# coding=utf-8
"""
Program:
Description:
Author: Lingyong Yan
Date: 2019-06-30 23:03:12
Last modified: 2019-08-23 02:55:14
Python release: 3.6
Notes:
"""
import os
import numpy as np
import sklearn
from scipy.stats.distributions import entropy
from sklearn.semi_supervised import label_propagation

import json
import scipy.sparse as ss
# from utils.data_load import load_graph, load_labels, load_json
# from utils.data_load import make_numpy

n_labeled_points = 10
max_iteration = 5


EOS = '</s>'


def rbf_kernel_safe(X, Y=None, gamma=None):

    X, Y = sklearn.metrics.pairwise.check_pairwise_arrays(X, Y)
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    K = sklearn.metrics.pairwise.euclidean_distances(X, Y, squared=True)
    K *= -gamma
    K -= K.max()
    np.exp(K, K)    # exponentiate K in-place
    return K


class NodeSet(object):
    def __init__(self, initial_nodes=[(EOS, 0)]):
        self.node2id = {}
        self.id2node = []
        self.counts = []
        for (node, count) in initial_nodes:
            self.add(node, count=count)

    def size(self):
        """
        @description: return the number of node in set
        @param:
        @return: int
        """
        return len(self.id2node)

    def add(self, node, count=1):
        """
        @description: add node (entity word or context pattern to set)
        @param {node:string}
        @return: None
        """
        if node not in self.node2id:
            self.node2id[node] = len(self.id2node)
            self.id2node.append(node)
            self.counts.append(0)
        node_id = self.node2id[node]
        self.counts[node_id] += count
        return node_id

    def get_id(self, node):
        return self.node2id.get(node)

    def get_node(self, _id):
        return self.id2node[_id]

    def get_count(self, _id):
        return self.counts[_id]

    def __len__(self):
        return len(self.id2node)

    def __contains__(self, item):
        return item in self.node2id

    def __getitem__(self, item):
        return self.node2id[item]

    @classmethod
    def from_file(cls, file_name):
        node_set = NodeSet()
        with open(file_name, 'r') as f:
            for line in f:
                [word, count] = line.strip().split('\t')
                node_set.add(word, count=int(count))
        return node_set


class Edges(object):
    def __init__(self, initial_edges=[], reverse=False):
        self.links = {}
        for (n_from, n_to, count) in initial_edges:
            self.add(n_from, n_to, count=count, reverse=reverse)

    def add(self, n_from, n_to, count=1, reverse=False):
        if reverse:
            temp = n_from
            n_from = n_to
            n_to = temp
        if n_from not in self.links:
            self.links[n_from] = {}
        self.links[n_from][n_to] = self.links[n_from].get(n_to, 0) + count


class BootstrappingGraph(object):
    def __init__(self, entities, patterns, edges):
        self.entities = entities
        self.patterns = patterns
        self.edges = edges

    def build_adjacency(self):
        """
        @description: build the sparse matrix of adjacency edges
                      of entity and patterns.
        @param:
        @return: sparse_matrix
        """
        rows, cols, vs = [], [], []
        for key, values in self.edges.links.items():
            for k, v in values.items():
                rows.append(key)
                cols.append(k)
                vs.append(v)
        rows = np.array(rows)
        cols = np.array(cols)
        vs = np.array(vs)
        sparse_m = ss.coo_matrix(
            (vs, (rows, cols)), shape=(len(self.entities), len(self.patterns)))
        return sparse_m

    @classmethod
    def from_file(cls, entity_file, pattern_file, link_file, skip_first=False):
        labels, links = [], []
        entities = NodeSet.from_file(entity_file)
        patterns = NodeSet.from_file(pattern_file)
        first_pos = 0 if skip_first else 1

        with open(link_file, 'r') as f:
            word_counts = {}
            for line in f:
                values = line.strip().split('\t')
                if not skip_first:
                    labels.append(values[0])
                if values[first_pos] not in word_counts:
                    word_counts[values[first_pos]] = 1
                else:
                    word_counts[values[first_pos]] += 1
                node_id = entities.get_id(values[first_pos])
                if node_id:
                    links.extend([(node_id, patterns[p], 1)
                                  for p in values[first_pos + 1:]
                                  if p in patterns])
            total_words = sum(word_counts.values())
            print("total_words:", total_words)
        edges = Edges(links)
        return cls(entities, patterns, edges)


def load_graph(entity_file, pattern_file, link_file):
    graph = BootstrappingGraph.from_file(entity_file, pattern_file, link_file)
    return graph


def load_truth(_file):
    with open(_file, 'r') as f:
        seeds = json.load(f)
        labels = list(seeds.keys())
    return seeds, labels


def pure_truth(ground_truth, seeds):
    pure_ground_truth = {}
    for key in ground_truth.keys():
        ss = set(ground_truth[key])
        pure_ground_truth[key] = ss - set(seeds[key])
    return pure_ground_truth


def make_numpy(graph):
    adj_mat = graph.build_adjacency()
    return adj_mat.todense()


def load_labels(fiile_name, vocab):
    counts, entities, labels = read_gold_mentions(fiile_name)
    entities, labels = majority_gold(counts, entities, labels)
    labels = np.insert(labels, 0, "NONE")
    indices = [0] + [vocab.get_id(e) for e in entities]
    return labels[indices]


def make_annotation(seeds, vocab, labels):
    n = vocab.size()
    y = -np.ones(n)
    for label in seeds:
        i = labels.index(label)
        for entity in seeds[label]:
            y[vocab.get_id(entity)] = i
    return y


def read_gold_mentions(filename):
    counts, entities, labels = [], [], []
    with open(filename) as f:
        for line in f:
            [entity, label, count] = line.strip().split('\t')
            counts.append(float(count))
            entities.append(entity)
            labels.append(label)
    return np.array(counts), np.array(entities), np.array(labels)


def majority_gold(counts, entities, labels):
    unique_entities = np.unique(entities)
    unique_labels = []
    for entity in unique_entities:
        entity_counts = counts[entities == entity]
        entity_labels = labels[entities == entity]
        max_label = entity_labels[np.argmax(entity_counts)]
        unique_labels.append(max_label)
    return unique_entities, np.array(unique_labels)


def annotate(seeds, vocab, labels):
    n = vocab.size()
    y = -np.ones(n)
    for label in seeds:
        i = labels.index(label)
        for entity in seeds[label]:
            y[vocab.get_id(entity)] = i
    return y


def train(X, y, y_train, labels, graph, output_file, top_n=100):
    indices = np.arange(len(y))
    total_promoted, epoch = 0, 1
    outputs = [['Epoch 0']]

    for _id, label in enumerate(labels):
        chunks = [label]
        current_output = []
        for i in indices[y_train == _id]:
            node = graph.entities.get_node(i)
            chunks.append(node)
            current_output.append(node)
            total_promoted += 1
        outputs.append([label] + current_output)
        print('\t'.join(chunks))

    print_state(epoch, labels, total_promoted, y_train)
    while total_promoted < 5522:  # 19875:  # 5522:
        outputs.append(['Epoch %d' % epoch])
        model = label_propagation.LabelPropagation(kernel='knn',
                                                   tol=0.01,
                                                   max_iter=2000,
                                                   n_jobs=16)
        model.fit(X, y_train)

        predictions = model.transduction_
        confidences = entropy(model.label_distributions_.T)

        for _id, label in enumerate(labels):
            mask = np.logical_and(predictions == _id, y_train == -1)
            ii = indices[mask]
            cc = confidences[mask]
            promoted = ii[np.argsort(cc)][:top_n]
            y_train[promoted] = _id
            chunks = [label]
            current_output = []
            for i in promoted:
                node = graph.entities.get_node(i)
                chunks.append(node)
                current_output.append(node)
            print('\t'.join(chunks))
            total_promoted += len(promoted)
            outputs.append([label] + current_output)
        print_state(epoch, labels, total_promoted, predictions)
        epoch += 1
    with open(output_file, 'w') as f:
        for line in pre_func(outputs):
            f.write(line+'\n')


def print_state(epoch, labels, total_promoted, predictions):
    print("Epoch %d --> size of chunks: %d" %
          (epoch, sum([len(c) for c in labels])))
    print('Epoch %d --> chunks: %s' % (epoch, str([c for c in labels])))
    print('Epoch %d --> total promoted: %d' % (epoch, total_promoted))
    unique, counts = np.unique(predictions, return_counts=True)
    print(dict(zip(unique, counts)))


def pre_func(datas):
    for data in datas:
        yield '\t'.join([str(d) for d in data])


if __name__ == "__main__":
    root = 'data/CoNLL'
    # root = 'data/OntoNotes'
    result_root = 'results/CoNLL'
    # result_root = 'results/OntoNotes'
    entity_file = os.path.join(root, 'entity_vocabulary.emboot.filtered.txt')
    pattern_file = os.path.join(root, 'pattern_vocabulary_emboot.filtered.txt')
    link_file = os.path.join(
        root, 'training_data_with_labels_emboot.filtered.txt')
    counts_file = os.path.join(root, 'entity_label_counts_emboot.filtered.txt')
    seeds_file = os.path.join(root, 'seedset.json')
    output_file = os.path.join(result_root, 'lp_results.txt')
    top_n = 100
    graph = load_graph(entity_file, pattern_file, link_file)
    X = make_numpy(graph)
    y = load_labels(counts_file, graph.entities)
    seeds, _ = load_truth(seeds_file)
    labels = list(seeds.keys())
    y_train = annotate(seeds, graph.entities, labels)
    train(X, y, y_train, labels, graph, output_file, top_n=top_n)
