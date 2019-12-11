# coding=UTF-8
"""
@Description:
@Author: Lingyong Yan
@Date: 2019-07-04 23:03:50
@LastEditTime: 2019-08-28 03:56:32
@LastEditors: Lingyong Yan
@Python release: 3.7
@Notes:
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .util import cosine_sim, masked_softmax
from .sparse import sparse_softmax_eff


class BootGCNLayer(nn.Module):
    def __init__(self, p2e_opt, e2p_opt, n_seed):
        super(BootGCNLayer, self).__init__()
        e2p_in, e2p_out = e2p_opt['in'], e2p_opt['out']
        self.e2p = HyperAttentionLayer(e2p_in, e2p_out, n_seed * e2p_in, e2p_in,
                                       50, sparse=e2p_opt['sparse'],
                                       disable_global=e2p_opt['disable_global'],
                                       alpha=0.5)
        p2e_in, p2e_out = p2e_opt['in'], p2e_opt['out']
        self.p2e = HyperAttentionLayer(p2e_in, p2e_out, n_seed * p2e_in, p2e_in,
                                       50, sparse=p2e_opt['sparse'],
                                       disable_global=p2e_opt['disable_global'],
                                       alpha=0.5)

    def reset_parameters(self):
        self.e2p.reset()
        self.p2e.reset()

    def forward(self, seeds, e_input, p_input, ep_adj, pe_adj):
        seeds = seeds.view(1, -1)
        p_output = self.e2p(seeds, p_input, e_input, e_input, pe_adj)
        e_output = self.p2e(seeds, e_input, p_output, p_output, ep_adj)
        '''
        e_num, p_num = e_input.size(0), p_input.size(0)
        e_step = e_num // 10 if e_num % 10 == 0 else e_num // 9
        p_step = p_num // 10 if p_num % 10 == 0 else p_num // 9
        if self.e2p.sparse:
            p_output, start = [], 0
            for adj in pe_adj:
                end = min(start + p_step, p_num)
                p_output.append(self.e2p(seeds, p_input[start:end], e_input, e_input, adj))
                start = end
            p_output = torch.cat(p_output, dim=0)
        else:
            p_output = self.e2p(seeds, p_input, e_input, e_input, pe_adj)

        if self.p2e.sparse:
            e_output, start = [], 0
            for adj in ep_adj:
                end = min(start + e_step, e_num)
                e_output.append(self.p2e(seeds, e_input[start:end], p_output, p_output, adj))
                start = end
            e_output = torch.cat(e_output, dim=0)
        else:
            e_output = self.p2e(seeds, e_input, p_output, p_output, ep_adj)
        '''
        return e_output, p_output


###############################################################################
# graph aggregation layers
###############################################################################

class GraphConvolution(nn.Module):
    def __init__(self, opt, gated=False, bias=True,
                 initializer=nn.init.kaiming_uniform_):
        super(GraphConvolution, self).__init__()
        self.opt = opt
        self.gated = gated
        self.bias = bias
        self.initializer = initializer
        self.in_features = opt['in']
        self.out_features = opt['out']

        self.linear = nn.Linear(self.in_features, self.out_features, bias=bias)

        if self.gated:
            self.gate = nn.Linear(self.in_features, 1, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):

        self.initializer(self.linear.weight)
        if self.gated:
            self.initializer(self.gate.weight)

    def forward(self, x, adj):
        if self.gated:
            gate_status = torch.sigmoid(self.gate(x))
            adj_hat = adj.t() * gate_status
            adj_hat = adj_hat.t()
        else:
            adj_hat = adj
        m = self.linear(x)
        output = torch.mm(adj_hat, m)
        return output


class LinearLayer(nn.Module):
    def __init__(self, opt, initializer=nn.init.xavier_uniform_, bias=True):
        super(LinearLayer, self).__init__()
        self.in_features = opt['in']
        self.out_features = opt['out']
        self.initializer = initializer

        self.linear = nn.Linear(self.in_features, self.out_features, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        self.initializer(self.linear.weight)

    def forward(self, x):
        return self.linear(x)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + '->' \
            + str(self.out_features) + ')'


class GraphAttentionLayer(nn.Module):
    def __init__(self, biased=True):
        super(GraphAttentionLayer, self).__init__()

    def forward(self, x, y, adj):
        cos = cosine_sim(x, y)

        mask = (1 - adj) * -1e10
        masked = cos + mask

        P = F.softmax(masked, dim=-1)
        output = 0.5 * torch.mm(P, y) + 0.5 * x
        return output


class HyperAttentionLayer(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 d_global,
                 d_local,
                 d_k,
                 alpha=0.5,
                 sparse=False,
                 disable_global=False,
                 dropout=0.5):
        super(HyperAttentionLayer, self).__init__()

        self.d_k = d_k
        self.d_global, self.d_local = d_global, d_local
        self.d_in, self.d_out = d_in, d_out
        self.alpha = alpha
        self.sparse = sparse

        self.w_ls = nn.Linear(d_local,  d_k)
        # self.w_gs = nn.Linear(d_global, d_k)
        self.w_ks = nn.Linear(d_in, d_k)
        self.w_vs = nn.Linear(d_in, d_out)
        self.w_res = nn.Linear(d_in, d_out)
        self.disable_global = disable_global
        if not self.disable_global:
            self.w_gs = nn.Linear(d_global, d_k)

        self.global_attn = ScaledDotProduct(temperature=np.power(d_k, 0.5))
        self.local_attn = ScaledDotProduct(temperature=np.power(d_k, 0.5))
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_out)
        self.reset()

    def reset(self):
        nn.init.normal_(self.w_ls.weight, mean=0,
                        std=np.sqrt(2.0 / (self.d_local + self.d_k)))
        if not self.disable_global:
            nn.init.normal_(self.w_gs.weight, mean=0,
                            std=np.sqrt(2.0 / (self.d_global + self.d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0,
                        std=np.sqrt(2.0 / (self.d_in + self.d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0,
                        std=np.sqrt(2.0 / (self.d_in + self.d_out)))
        nn.init.normal_(self.w_res.weight, mean=0,
                        std=np.sqrt(2.0 / (self.d_in + self.d_out)))

    def forward(self, global_q, local_q, k, v, mask=None):
        len_lq, _ = local_q.size()

        residual = self.w_res(local_q)
        lq = self.w_ls(local_q)
        k = self.w_ks(k)
        v = self.w_vs(v)

        l_attn = self.local_attn(lq, k, mask, q_len=len_lq, sparse=self.sparse)
        if self.disable_global:
            attn = l_attn
        else:
            gq = self.w_gs(global_q)
            g_attn = self.global_attn(gq, k, mask, True, q_len=len_lq,
                                      sparse=self.sparse)
            attn = self.alpha * g_attn + (1 - self.alpha) * l_attn

        if self.sparse:
            output = torch.sparse.mm(attn, v)
        else:
            output = torch.mm(attn, v)
        output = self.dropout(output)
        output = self.layer_norm(F.relu(output + residual, inplace=True))

        return output


class ScaledDotProduct(nn.Module):
    def __init__(self, temperature, dropout=0.1):
        super(ScaledDotProduct, self).__init__()
        self.temperature = temperature
        # self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, mask=None, repeat=False, q_len=0, sparse=False):
        if sparse:
            indices = mask
            # indices = mask.indices()
            if repeat:
                q = q.repeat(indices.size(1), 1)
                attn = torch.sum(torch.mul(q, k[indices[1]]), dim=-1)
            else:
                attn = torch.sum(
                    torch.mul(q[indices[0]], k[indices[1]]), dim=-1)
            attn = attn / self.temperature
            attn = (attn - attn.max()).exp()
            attn = torch.sparse.FloatTensor(
                indices, attn, size=torch.Size((q_len, k.size(0))))
            # attn = attn.sparse_mask(mask)
            attn = sparse_softmax_eff(attn.coalesce())
        else:
            attn = torch.mm(q, k.transpose(-1, -2))
            attn = attn / self.temperature
            if repeat:
                attn = attn.repeat(q_len, 1)
            if mask is not None:
                attn = masked_softmax(attn, mask, memory_efficient=True)
            else:
                attn = F.softmax(attn, dim=-1)
        return attn
