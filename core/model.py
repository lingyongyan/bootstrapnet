# coding=UTF-8
"""
@Description:
@Author: Lingyong Yan
@Date: 2019-07-04 23:02:12
@LastEditTime: 2019-08-30 01:09:39
@LastEditors: Lingyong Yan
@Python release: 3.7
@Notes:
"""
import torch
import torch.nn as nn
from torch.nn.modules.rnn import GRUCell
from torch.sparse import mm as sparse_mm
import torch.nn.functional as F
from .layer import BootGCNLayer
from .util import group_select_eff, heuristic_pooling, cosine_sim, group_select


class model_check(object):
    def save(self, optimizer, filename):
        params = {
            'model': self.state_dict(),
            'optim': optimizer.state_dict()
        }
        try:
            print('print model to path:%s' % filename)
            torch.save(params, filename)
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    def load(self, optimizer, filename, device):
        try:
            print('load model from path:%s' % filename)
            checkpoint = torch.load(filename, map_location=device)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optim'])
        return optimizer


class BootTeacher(nn.Module, model_check):
    def __init__(self, opt):
        super(BootTeacher, self).__init__()
        self.sparse = opt['sparse']
        self.encoder = BootGCN(opt)
        self.classifier = BootClassifier(opt)

    def forward(self, seeds, es, ps, ep_adj, pe_adj):
        n_es, n_ps = self.encoder(seeds, es, ps, ep_adj, pe_adj)
        output = self.classifier(n_es)
        return output, n_es, n_ps


class BootNet(nn.Module, model_check):
    def __init__(self, opt):
        super(BootNet, self).__init__()
        self.sparse = opt['sparse']
        self.encoder = BootGCN(opt)
        if opt['no_boot']:
            self.decoder = BootExpander(opt)
        else:
            self.decoder = BootDecoder(opt)

    def forward(self, seeds, es, ps, ep_adj, pe_adj, neighbors, dev=None):
        es, ps = self.encoder(seeds, es, ps, ep_adj, pe_adj)
        es = es.detach()
        outputs = self.decoder(seeds, es, neighbors, dev=dev)
        return outputs


class BootGCN(nn.Module):
    def __init__(self, opt):
        super(BootGCN, self).__init__()
        self.opt = opt
        self.iter = opt['layer']
        self.dropout = opt['dropout']
        self.layers = nn.ModuleList()
        self.seed_total = opt['seed_count'] * opt['num_class']
        for i in range(self.iter):
            in_feature = opt['num_feature']
            out_feature = opt['num_feature']
            opt_ = dict([('in', in_feature), ('out', out_feature)])
            opt_['sparse'] = opt['sparse']
            opt_['disable_global'] = opt['disable_global']
            gcn_layer = BootGCNLayer(opt_, opt_, self.seed_total)
            self.layers.append(gcn_layer)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, seed_index, es, ps, ep_adj, pe_adj):
        for layer in self.layers:
            seeds = es[seed_index]
            es, ps = layer(seeds, es, ps, ep_adj, pe_adj)
        return es, ps


class BootClassifier(nn.Module):
    def __init__(self, opt):
        super(BootClassifier, self).__init__()
        self.opt = opt
        in_feature = opt['num_feature']
        out_feature = opt['num_class']
        self.dropout = opt['dropout']
        self.fc = nn.Linear(in_feature, out_feature)
        self.reset()

    def reset(self):
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, es):
        es = F.dropout(es, self.dropout, training=self.training)
        x = self.fc(es)
        return x


class BootExpander(nn.Module):
    def __init__(self, opt):
        super(BootExpander, self).__init__()
        self.opt = opt
        self.step = opt['rnn_step']
        self.seed_count = opt['seed_count']
        self.with_update = opt['with_update']
        self.ave_method = opt['ave_method']
        self.min_match = max(1, opt['min_match'])
        self.n_class, self.n_feature = opt['num_class'], opt['num_feature']

    def forward(self, seeds, es, neighbors, dev=None):
        outputs = []
        selects = []
        steps = []
        device = seeds.device
        entity_mask = torch.zeros(es.size(0), dtype=torch.uint8, device=device)
        entity_mask.scatter_(0, seeds, 1)
        seed_mask = torch.zeros(es.size(0), dtype=torch.uint8, device=device)
        seed_mask.scatter_(0, seeds, 1)
        select_num = self.seed_count
        cate_group = []
        for start in range(0, seeds.size(0), self.seed_count):
            end = min(seeds.size(0),  start + self.seed_count)
            cate_mask = torch.zeros(es.size(0), dtype=torch.uint8, device=device)
            cate_mask.scatter_(0, seeds[start:end], 1)
            cate_group.append(cate_mask)
        min_match = self.min_match
        for rnn_i in range(self.step):
            if rnn_i > 2:
                min_match = 2
            else:
                min_match = max(2, self.min_match - rnn_i)
            if self.with_update:
                category_selected = torch.where(entity_mask)[0]
                category_size = [g.sum().long() for g in cate_group]
            else:
                category_selected = torch.where(seed_mask)[0]
                category_size = self.seed_count
            categories = heuristic_pooling(
                es[category_selected], category_size, method=self.ave_method)

            cate_valid = []
            cate_pools = 0.
            for cate_mask in cate_group:
                cate_entity = (cate_mask).unsqueeze(-1).float()
                if neighbors.is_sparse:
                    cate_pool = sparse_mm(neighbors, cate_entity) > min_match
                else:
                    cate_pool = torch.mm(neighbors, cate_entity) > min_match
                cate_pool = cate_pool.squeeze(-1)
                cate_pool[entity_mask] = 0
                cate_valid.append(cate_pool)
                cate_pools += cate_pool.float()
            cate_pools = cate_pools > 0.
            # pss = torch.mean(cate_pools.eq(entity_pool).float())
            # assert pss == 1.0
            sims = cosine_sim(es[cate_pools], categories) * 0.5 + 0.5
            probs = torch.zeros((es.size(0), self.n_class),
                                device=device).type_as(sims)
            probs[cate_pools] = sims

            group = group_select_eff(probs.detach(), cate_valid, cate_pools,
                                     top_n=select_num, training=self.training)
            # group = group_select(probs.detach(), cate_pools, top_n=select_num, training=self.training)
            for c in range(len(cate_group)):
                cate_group[c].scatter_(0, group[c], 1)
            assert len(group) == len(cate_group)
            step = [g.size(0) for g in group]
            last_selected = torch.cat(group, dim=0)
            entity_mask.scatter_(0, last_selected, 1)
            outputs.append(probs[last_selected])
            selects.append(last_selected)
            steps.append(step)
        return outputs, selects, steps, []


class BootDecoder(nn.Module):
    def __init__(self, opt):
        super(BootDecoder, self).__init__()
        self.opt = opt
        self.rnn_step = opt['rnn_step']
        self.dropout = opt['dropout']
        self.seed_count = opt['seed_count']
        self.ave_method = opt['ave_method']
        self.min_match = max(1, opt['min_match'])
        self.layers = nn.ModuleList()
        self.n_class, self.n_feature = opt['num_class'], opt['num_feature']
        self.rnn_cell = GRUCell(self.n_feature, self.n_feature)
        self.layer_norm = nn.LayerNorm(self.n_feature)
        self.dev = False

    def forward(self, seeds, es, neighbors, hx=None, dev=None):
        outputs = []
        selects = []
        steps = []
        device = seeds.device
        if self.dev:
            entity_mask = torch.ones(es.size(0), dtype=torch.uint8, device=device)
            entity_mask.scatter_(0, dev, 0)
            select_num = self.seed_count // 2
        else:
            entity_mask = torch.zeros(es.size(0), dtype=torch.uint8, device=device)
            entity_mask.scatter_(0, seeds, 1)
            if dev is not None:
                entity_mask.scatter_(0, dev, 1)
            select_num = self.seed_count
        # neighbors = neighbor_adj(adj)
        last_selected = seeds
        step = self.seed_count
        rnn_step = self.rnn_step # if self.training else self.rnn_step * 2 + 5
        output_hx = []
        cate_group = []
        for start in range(0, seeds.size(0), self.seed_count):
            end = min(seeds.size(0),  start + self.seed_count)
            cate_mask = torch.zeros(es.size(0), dtype=torch.uint8, device=device)
            cate_mask.scatter_(0, seeds[start:end], 1)
            cate_group.append(cate_mask)
        min_match = self.min_match
        for rnn_i in range(rnn_step):
            if rnn_i > 2:
                min_match = 2
            else:
                min_match = max(2, self.min_match - rnn_i)
            if torch.sum((entity_mask == 0).float()) == 0:
                continue
            inp = heuristic_pooling(es[last_selected], step, self.ave_method)
            hx = self.rnn_cell(inp, hx)
            output_hx.append(hx)

            '''
            current_entity = (entity_mask).unsqueeze(-1).float()
            if neighbors.is_sparse:
                entity_pool = sparse_mm(neighbors, current_entity) > 1
            else:
                entity_pool = torch.mm(neighbors, current_entity) > 1
            entity_pool = entity_pool.squeeze(-1)
            entity_pool[entity_mask] = 0
            '''
            if not self.dev:
                cate_valid = []
                cate_pools = 0.
                for cate_mask in cate_group:
                    cate_entity = (cate_mask).unsqueeze(-1).float()
                    if neighbors.is_sparse:
                        cate_pool = sparse_mm(neighbors, cate_entity) >= min_match
                    else:
                        cate_pool = torch.mm(neighbors, cate_entity) >= min_match
                    cate_pool = cate_pool.squeeze(-1)
                    cate_pool[entity_mask] = 0
                    cate_valid.append(cate_pool)
                    cate_pools += cate_pool.float()
                cate_pools = cate_pools > 0.
                # pss = torch.mean(cate_pools.eq(entity_pool).float())
                # assert pss == 1.0
                sims = cosine_sim(es[cate_pools], hx) * 0.5 + 0.5
                probs = torch.zeros((es.size(0), self.n_class),
                                    device=device).type_as(sims)
                probs[cate_pools] = sims
                # group = group_select(probs.detach(), cate_pools, top_n=select_num, training=self.training)
                group = group_select_eff(probs.detach(), cate_valid, cate_pools,
                                         top_n=select_num, training=self.training)
            else:
                entity_pools = (~entity_mask)
                sims = cosine_sim(es[entity_pools], hx) * 0.5 + 0.5
                probs = torch.zeros((es.size(0), self.n_class),
                                    device=device).type_as(sims)
                probs[entity_pools] = sims
                group = group_select(probs.detach(), entity_pools,
                                     top_n=select_num, training=self.training)
            for c in range(len(cate_group)):
                cate_group[c].scatter_(0, group[c], 1)
            step = [g.size(0) for g in group]
            last_selected = torch.cat(group, dim=0)
            entity_mask.scatter_(0, last_selected, 1)
            outputs.append(probs[last_selected])
            selects.append(last_selected)
            steps.append(step)
        return outputs, selects, steps, output_hx
