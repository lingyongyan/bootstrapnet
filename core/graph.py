# coding=UTF-8
"""
@Description:
@Author: Lingyong Yan
@Date: 2019-07-24 01:00:10
@LastEditTime: 2019-08-28 08:20:17
@LastEditors: Lingyong Yan
@Python release: 3.7
@Notes:
"""
import torch
import numpy as np
import math

PAD = '<pad>'


class BootGraph(object):
    def __init__(self, vocab_e, vocab_p):
        self.vocab_e = vocab_e
        self.vocab_p = vocab_p
        self.edges = []

        self.node_size = -1

    def add(self, e_string, p_string, w):
        if w > 0 and e_string in self.vocab_e.stoi and \
          p_string in self.vocab_p.stoi:
            e = self.vocab_e.stoi[e_string]
            p = self.vocab_p.stoi[p_string]
            self.edges.append((e, p, w))

    def to_symmetric(self, self_link_weight=1.0):
        vocab = set()
        for u, v, w in self.edges:
            vocab.add(u)
            vocab.add(v)

        pair2wt = dict()
        for u, v, w in self.edges:
            pair2wt[(u, v)] = w

        edges_ = list()
        for (u, v), w in pair2wt.items():
            if u == v:
                continue
            w_ = pair2wt.get((v, u), -1)
            if w > w_:
                edges_ += [(u, v, w), (v, u, w)]
            elif w == w_:
                edges_ += [(u, v, w)]
        for k in vocab:
            edges_ += [(k, k, self_link_weight)]

        d = dict()
        for u, v, w in edges_:
            d[u] = d.get(u, 0.0) + w

        self.edges = [(u, v, w / math.sqrt(d[u] * d[v])) for u, v, w in edges_]

    def get_adjacency(self, device=None, sparse=False):
        shape = torch.Size([self.vocab_e.vocab_size, self.vocab_p.vocab_size])

        us, vs, ws = [], [], []
        for u, v, w in self.edges:
            us += [u]
            vs += [v]
            ws += [w]
        index = torch.LongTensor([us, vs])
        value = torch.FloatTensor(ws)
        adj = torch.sparse.FloatTensor(index, value, shape)
        if not sparse:
            adj = adj.to_dense()
        if device is not None:
            adj = adj.to(device)

        return adj

    @classmethod
    def from_file(cls, file_name, node, split='\t', weight=None):
        vocab_e, vocab_p, col_e, col_p = node
        col_w = weight

        graph = cls(vocab_e, vocab_p)

        with open(file_name, 'r') as f:
            for line in f:
                items = line.strip().split(split)
                se, sp = items[col_e], items[col_p]
                sw = items[col_w] if col_w else None
                w = float(sw) if sw else 1
                graph.add(se, sp, w)
        return graph


class Vocab(object):
    def __init__(self, with_padding=False):
        self.itos = []
        self.stoi = {}
        self.vocab_size = 0

        if with_padding:
            self.add(PAD)

    def add(self, string):
        if string not in self.stoi:
            self.stoi[string] = self.vocab_size
            self.itos.append(string)
            self.vocab_size += 1

    def __len__(self):
        return self.vocab_size

    @classmethod
    def from_file(cls, file_name, cols, split='\t', with_padding=False):
        vocab = cls(with_padding)
        with open(file_name, 'r') as f:
            for line in f:
                items = line.strip().split(split)
                for col in cols:
                    item = items[col]
                    for string in item.strip().split(' '):
                        string = string.split(":")[0]
                        vocab.add(string)
        return vocab


class EntityLabel(object):
    def __init__(self, vocab_size):
        self.itol = [-1 for k in range(vocab_size)]

    def add(self, n, l):
        self.itol[n] = l

    @classmethod
    def from_file(cls, file_name, node, label, split='\t'):
        vocab_n, col_n = node
        vocab_l, col_l = label

        entity_label = cls(len(vocab_n))
        with open(file_name, 'r') as f:
            for line in f:
                items = line.strip().split(split)
                sn, sl = items[col_n], items[col_l]
                sl = sl.strip().split(' ')[0]
                n = vocab_n.stoi.get(sn, -1)
                label = vocab_l.stoi.get(sl, -1)
                if n != -1:
                    entity_label.add(n, label)
        return entity_label


class MultiEntityLabel(object):
    def __init__(self, vocab_size):
        self.itol = [[] for k in range(vocab_size)]

    def add(self, n, l):
        if l not in self.itol[n]:
            self.itol[n].append(l)

    def to_tensor(self, label_size):
        tensor = torch.zeros((len(self.itol), label_size), dtype=torch.long)
        for n, labels in enumerate(self.itol):
            tensor[n, labels] = 1
        return tensor

    @classmethod
    def from_file(cls, file_name, node, label, split='\t'):
        vocab_n, col_n = node
        vocab_l, col_l = label

        entity_label = cls(len(vocab_n))
        with open(file_name, 'r') as f:
            for line in f:
                items = line.strip().split(split)
                sn, sls = items[col_n], items[col_l]
                sls = sls.strip().split(' ')
                n = vocab_n.stoi.get(sn, -1)
                if n != -1:
                    for sl in sls:
                        label = vocab_l.stoi.get(sl, -1)
                        if label != -1:
                            entity_label.add(n, label)
        return entity_label


class Feature(object):
    def __init__(self, vocab_size, feature_size):
        self.itof = np.zeros((vocab_size, feature_size))
        self.vocab_size = vocab_size
        self.feature_size = feature_size
        self.one_hot = None

    def to_one_hot(self, binary=False, normalize=True):
        self.one_hot = self.itof.copy()
        if binary:
            self.one_hot[self.one_hot > 0] = 1.0

        if normalize:
            self.one_hot = self.one_hot / self.one_hot.sum(axis=1,
                                                           keepdims=True)

    @classmethod
    def from_file(cls, file_name, node, feature, link=True, split='\t'):
        vocab_n, col_n = node
        if link:
            vocab_f, col_f = feature
            f_size = len(vocab_f)
        else:
            f_size, col_f = feature

        entity_feature = cls(len(vocab_n), f_size)
        with open(file_name, 'r') as f:
            for idx, line in enumerate(f):
                items = line.strip().split(split)
                sn, sf = items[col_n], items[col_f]
                n = vocab_n.stoi.get(sn, -1)
                if n == -1:
                    continue
                if link:
                    fs, ws = [], []
                    for string in sf.strip().split(' '):
                        f, w = string.split(':')
                        f, w = vocab_f.stoi.get(f, -1), float(w)
                        if f != -1:
                            fs.append(f)
                            ws.append(w)
                    entity_feature.itof[n, f] = w
                else:
                    ws = sf.split(' ')
                    ws = [float(w) for w in ws]
                    entity_feature.itof[n] = ws
        return entity_feature
