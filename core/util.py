# coding=UTF-8
"""
@Description:
@Author: Lingyong Yan
@Date: 2019-07-05 01:39:54
@LastEditTime: 2019-08-30 01:08:46
@LastEditors: Lingyong Yan
@Python release: 3.7
@Notes:
"""
import os
import logging
from itertools import product, combinations

import numpy as np
import torch
from torchtext.data import Field, Dataset, Example


def noisy_or(probs, adj):
    log_rev_probs = (1. - probs.clamp(max=1-1e-5)).log()
    log_rev_entity_probs = torch.mm(adj, log_rev_probs)
    entity_probs = 1. - log_rev_entity_probs.exp()
    return entity_probs


def group_select_eff(probs, cate_valid, valid, top_n=100, training=False):
    group = []
    new_probs = torch.zeros_like(probs, device=probs.device)
    for c, cate_mask in enumerate(cate_valid):
        new_probs[cate_mask, c] = probs[cate_mask, c]
    new_probs = new_probs / (new_probs.sum(dim=-1, keepdim=True) + 1e-16)
    index = torch.full((probs.size(0),), -1, dtype=torch.long, device=probs.device)
    if training:
        index[valid] = torch.multinomial(new_probs[valid], 1).squeeze(-1)
    else:
        index[valid] = new_probs[valid].argmax(dim=-1)
    group = []
    for i in range(probs.size(-1)):
        ii = (valid & (index == i))
        group_index = torch.where(ii)[0]
        select_top = torch.argsort(probs[ii, i], descending=True)[:top_n]
        select_index = group_index[select_top]
        group.append(select_index.view(-1).detach())
    return group


def group_select(probs, mask, top_n=100, training=False):
    valid = mask > 0.
    index = torch.full((probs.size(0),), -1, dtype=torch.long, device=probs.device)
    entropy = torch.full((probs.size(0),), 1e10, dtype=torch.float, device=probs.device)
    if training:
        index[valid] = torch.multinomial(probs[valid], 1).squeeze(-1)
    else:
        index[valid] = probs[valid].argmax(dim=-1)
    entropy[valid] = torch.sum(- probs[valid] * probs[valid].log(), dim=-1)
    group = []
    for i in range(probs.size(-1)):
        ii = (mask & (index == i))
        group_index = torch.where(ii)[0]
        select_top = torch.argsort(entropy[ii])[:top_n]
        select_index = group_index[select_top]
        group.append(select_index.view(-1).detach())
    return group


def cosine_sim(a, b, dim=-1, eps=1e-8):
    """calculate the cosine similarity and avoid the zero-division
    """
    a_norm = a / (a.norm(dim=dim)[:, None]).clamp(min=eps)
    b_norm = b / (b.norm(dim=dim)[:, None]).clamp(min=eps)
    if len(a.shape) <= 2:
        sim = torch.mm(a_norm, b_norm.transpose(1, 0))
    else:
        sim = torch.einsum('ijk, lmk->iljm', (a_norm, b_norm))
    return sim


def inner_product(a, b, eps=1e-8):
    """calculate the inner product of two vectors
    """
    if len(a.shape) <= 2:
        sim = torch.mm(a, b.t())
    else:
        sim = torch.einsum('ijk, lmk->iljm', (a, b))
    return sim


def get_optimizer(name, parameters, lr, weight_decay=0):
    """initialize parameter optimizer
    """
    if name == 'sgd':
        return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'rmsprop':
        return torch.optim.RMSprop(parameters, lr=lr,
                                   weight_decay=weight_decay)
    elif name == 'adagrad':
        return torch.optim.Adagrad(parameters, lr=lr,
                                   weight_decay=weight_decay)
    elif name == 'adam':
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adamax':
        return torch.optim.Adamax(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise Exception("Unsupported optimizer: {}".format(name))


def change_lr(optimizer, new_lr):
    """change the learing rate in the optimizer
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def group_mask(group_num, num_per_group, device=None):
    total = group_num * num_per_group
    pos_mask = torch.zeros((total, total), dtype=torch.bool, device=device)
    neg_mask = torch.zeros_like(pos_mask)

    for i in range(group_num):
        offset = i * num_per_group
        for j, k in combinations(range(num_per_group), 2):
            pos_mask[offset+j][offset+k] = 1

    for i, j in combinations(range(group_num), 2):
        o_i = i * num_per_group
        o_j = j * num_per_group
        for k, l in product(range(num_per_group), range(num_per_group)):
            neg_mask[o_i + k][o_j + l] = 1
    return pos_mask, neg_mask


def class_mask(class_num, device=None):
    neg_mask = torch.zeros((class_num, class_num), dtype=torch.bool, device=device)

    for i, j in combinations(range(class_num), 2):
        neg_mask[i][j] = 1

    return neg_mask


def neighbor_mask(input_adj, device, length=4, file_path=None):
    if file_path and os.path.exists(file_path+'/neg_mask.pt'):
        ret_pos_mask = torch.load(file_path + '/pos_mask.pt')
        ret_neg_mask = torch.load(file_path + '/neg_mask.pt')
        ret_pos_mask = ret_pos_mask.to(device)
        ret_neg_mask = ret_neg_mask.to(device)
    else:
        sparse = input_adj.is_sparse
        if sparse:
            adj = input_adj.to_dense()
        else:
            adj = input_adj
        adj = adj > 0
        pos_size = adj.size(0) * 2
        pos_select = torch.argsort(torch.sum(adj, dim=0))[:pos_size]
        pos_mask = torch.zeros_like(adj).type_as(adj)
        pos_mask[:, pos_select] = adj[:, pos_select]
        neg_mask = adj
        adj = adj.float()
        for i in range(length-1):
            neg_mask = torch.mm(neg_mask.float(), adj.t()) > 0
            neg_mask = torch.mm(neg_mask.float(), adj) > 0
        neg_mask = ~ neg_mask
        if sparse:
            ret_pos_mask = torch.nonzero(pos_mask.cpu()).t()
            ret_neg_mask = torch.nonzero(neg_mask.cpu()).t()
            length = ret_neg_mask.size(1)
            selected = np.random.choice(range(length), size=length//5, replace=False)
            ret_neg_mask = ret_neg_mask[:, selected]
            del adj
            del pos_mask
            del neg_mask
        else:
            ret_pos_mask = pos_mask
            ret_neg_mask = neg_mask
        if file_path:
            torch.save(ret_pos_mask.cpu(), file_path + '/pos_mask.pt')
            torch.save(ret_neg_mask.cpu(), file_path + '/neg_mask.pt')
    # ret_pos_mask, ret_neg_mask
    return ret_pos_mask, ret_neg_mask


def neighbor_adj(adj, device, length=1, file_path=None):
    if file_path and os.path.exists(os.path.join(file_path, 'non_neighbor.pt')):
        neighbor = torch.load(os.path.join(file_path, 'neighbor.pt'))
        if isinstance(neighbor, tuple):
            neighbor = torch.sparse.FloatTensor(neighbor[0], neighbor[1], neighbor[2])
        non_neighbor = torch.load(os.path.join(file_path, 'non_neighbor.pt'))
        non_neighbor = non_neighbor.to(device)
    else:
        sparse = adj.is_sparse
        if sparse:
            adj = adj.to_dense()
        adj = adj > 0
        neighbor = torch.mm(adj.float(), adj.float().t()) > 0
        non_neighbor = neighbor
        for i in range(1, length):
            non_neighbor = torch.mm(non_neighbor.float(), adj.float()) > 0
            non_neighbor = torch.mm(non_neighbor.float(), adj.float().t()) > 0
        non_neighbor = ~ non_neighbor
        if sparse:
            neighbor = neighbor.to_sparse().detach()
            # non_neighbor = non_neighbor.to_sparse().detach()
        neighbor = neighbor.float().cpu()
        non_neighbor = non_neighbor.cpu()
        if file_path:
            if sparse:
                indices, values = neighbor.indices(), neighbor.values()
                size = neighbor.size()
                save_tuple = (indices, values, size)
                torch.save(save_tuple, os.path.join(file_path, 'neighbor.pt'))
                non_neighbor = torch.nonzero(non_neighbor.cpu()).t()
                non_length = non_neighbor.size(1)
                selected = np.random.choice(range(non_length), size=non_length//5, replace=False)
                non_neighbor = non_neighbor[:, selected]
                torch.save(non_neighbor, os.path.join(file_path, 'non_neighbor.pt'))
            else:
                torch.save(neighbor, os.path.join(file_path, 'neighbor.pt'))
                torch.save(non_neighbor, os.path.join(file_path, 'non_neighbor.pt'))
    return neighbor, non_neighbor


def bidirection_adj(adj, file_path=None):
    if file_path and os.path.exists(os.path.join(file_path, 'ep_adj.pt')):
        adj = torch.load(os.path.join(file_path, 'ep_adj.pt'))
        t_adj = torch.load(os.path.join(file_path, 'pe_adj.pt'))
    else:
        if adj.is_sparse:
            t_adj = adj.t().coalesce().indices()
            adj = adj.coalesce().indices()
        else:
            t_adj = adj.t()
        if file_path:
            torch.save(adj.cpu(), file_path + '/pos_mask.pt')
            torch.save(t_adj.cpu(), file_path + '/neg_mask.pt')
    return adj, t_adj


def bidirection_adj_eff(adj, file_path=None):
    if file_path and os.path.exists(os.path.join(file_path, 'ep_adj.pt')):
        adj = torch.load(os.path.join(file_path, 'ep_adj.pt'))
        t_adj = torch.load(os.path.join(file_path, 'pe_adj.pt'))
    else:
        if adj.is_sparse:
            t_adj = adj.t().coalesce().cpu().indices()
            adj = adj.coalesce().cpu().indices()
        else:
            t_adj = adj.t()
        if file_path:
            torch.save(adj.cpu(), file_path + '/ep_adj.pt')
            torch.save(t_adj.cpu(), file_path + '/pe_adj.pt')
    return adj, t_adj


def remove_wild_char(word_list):
    return [w for w in word_list if w != '_']


class PatternField(Field):
    def __init__(self, *args, **kwargs):
        logger = logging.getLogger(__name__)
        if kwargs.get('batch_first') is False:
            logger.warning("Option batch_first has to be set to "
                           "use pytorch-seq2seq.  Changed to True.")
        kwargs['batch_first'] = True
        if kwargs.get('include_lengths') is False:
            logger.warning("Option include_lengths has to be set "
                           "to use pytorch-seq2seq.  Changed to True.")
        kwargs['include_lengths'] = True
        if kwargs.get('lower') is True:
            logger.warning("Option lower has to be set to use "
                           "pretrained embedding.  Changed to False.")
        kwargs['lower'] = True
        kwargs['preprocessing'] = remove_wild_char

        super(PatternField, self).__init__(*args, **kwargs)

    def build_vocab(self, word_vectors=None, itos=None):
        assert word_vectors is not None or itos is not None
        if word_vectors is not None:
            words = [word_vectors.itos]
            super(PatternField, self).build_vocab(words, vectors=word_vectors)
        else:
            words = [itos]
            super(PatternField, self).build_vocab(words)


class PatternDataset(Dataset):
    def __init__(self, pattern_list, fields, **kwargs):
        examples = []
        for pattern in pattern_list:
            if not isinstance(pattern, str):
                pattern = str(pattern)
            example = Example.fromlist([pattern], fields)
            examples.append(example)
        if isinstance(fields, dict):
            fields, field_dict = [], fields
            for field in field_dict.values():
                if isinstance(field, list):
                    fields.extend(field)
                else:
                    fields.append(field)
        super(PatternDataset, self).__init__(examples, fields, **kwargs)


def sequence_mask(lens, max_len=None):
    """get a mask matrix from batch lens tensor

    :param lens:
    :param max_len:  (Default value = None)

    """
    if max_len is None:
        max_len = lens.max().item()
    batch_size = lens.size(0)

    ranges = torch.arange(0, max_len, device=lens.device).long()
    ranges = ranges.unsqueeze(0).expand(batch_size, max_len)
    lens_broadcast = lens.unsqueeze(-1).expand_as(ranges)
    mask = ranges < lens_broadcast
    return mask


def mask_mean_weights(mask):
    new_mask = mask.float()
    sum_mask = new_mask.sum(dim=1, keepdim=True)
    indice = (sum_mask > 0).squeeze(1)
    new_mask[indice] /= sum_mask[indice]
    return new_mask


def masked_softmax(vector: torch.Tensor,
                   mask: torch.Tensor,
                   dim: int = -1,
                   memory_efficient: bool = False,
                   mask_fill_value: float = -1e32) -> torch.Tensor:
    """
    This function is copied from allennlp.nn.util.py, which can perform masked softmax
    ``torch.nn.functional.softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular softmax.
    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    If ``memory_efficient`` is set to true, we will simply use a very large negative number for those
    masked positions so that the probabilities of those positions would be approximately 0.
    This is not accurate in math, but works for most cases and consumes less memory.
    In the case that the input vector is completely masked and ``memory_efficient`` is false, this function
    returns an array of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of
    a model that uses categorical cross-entropy loss. Instead, if ``memory_efficient`` is true, this function
    will treat every element as equal, and do softmax over equal numbers.
    """
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the mask, we zero these out.
            result = torch.nn.functional.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_vector = vector.masked_fill((1 - mask).byte(), mask_fill_value)
            result = torch.nn.functional.softmax(masked_vector, dim=dim)
    return result


def heuristic_pooling(x, step, method='average'):
    def torch_average(x, dim=-2):
        return torch.mean(x, dim)

    def torch_max(x, dim=-2):
        return torch.max(x, dim)[0]

    def torch_min(x, dim=-2):
        return torch.min(x, dim)[0]

    def check_step(x, step):
        if isinstance(step, list):
            return x.size(0) == sum(step)
        else:
            return x.size(0) % step == 0

    assert check_step(x, step)

    if method.lower() == 'max':
        func = torch_max
    elif method.lower() == 'min':
        func = torch_min
    else:
        func = torch_average

    if isinstance(step, list):
        output = []
        step_start = 0
        for s in step:
            step_end = step_start + s
            if s == 0:
                value = torch.zeros(x.size(-1), device=x.device).type_as(x)
            else:
                value = func(x[step_start:step_end])
            output.append(value)
            step_start = step_end
        return torch.stack(output, dim=0)
    else:
        x = x.view(-1, step, x.size(-1))
        output = func(x)
        return output


def predict_confidence(predicts):
    n_class = predicts.size(1)
    max_entropy = np.log(n_class)
    entropy = -torch.mean(predicts * predicts.log(), dim=-1)
    confidence = 1 - entropy / max_entropy
    return confidence
