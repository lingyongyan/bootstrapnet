# coding=UTF-8
"""
@Description:
@Author: Lingyong Yan
@Date: 2019-08-06 22:42:04
@LastEditTime: 2019-08-28 11:57:42
@LastEditors: Lingyong Yan
@Python release: 3.7
@Notes:
"""

import numpy as np
import random
import argparse
import torch
import os

from core.graph import Vocab, EntityLabel, Feature, BootGraph, MultiEntityLabel
from core.model import BootTeacher,  BootNet
from core.framework import boot_train, noboot_train
from core.util import get_optimizer


def regist_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='data/OntoNotes')
    parser.add_argument('--save', type=str, default='saved')
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--s_load', type=str, default=None)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--gold', default='')
    parser.add_argument('--optimizer', default='adam', help='Optimizer')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--decay', type=float, default=1e-3,
                        help='Weight decay for optimization')
    parser.add_argument('--device', type=int, default=5)
    parser.add_argument('--bert', action='store_true',
                        help='Using Bert features')
    parser.add_argument('--s_device', type=int, default=2)
    parser.add_argument('--t_prepre_epoch', type=int, default=0)
    parser.add_argument('--t_pre_epoch', type=int, default=200)
    parser.add_argument('--s_pre_epoch', type=int, default=100)
    parser.add_argument('--t_epoch', type=int, default=50)
    parser.add_argument('--s_epoch', type=int, default=50)
    parser.add_argument('--t_draw', default='max')
    parser.add_argument('--s_draw', default='smp')
    parser.add_argument('--uns_scale', type=int, default=1)
    parser.add_argument('--nn_scale', type=float, default=1.0)
    parser.add_argument('--ave_method', default='average')
    parser.add_argument('--initialization', default='embedding')
    parser.add_argument('--iter', type=int, default=10)
    parser.add_argument('--save_step', type=int, default=10)
    parser.add_argument('--rnn_step', type=int, default=20)
    parser.add_argument('--min_match', type=int, default=2)
    parser.add_argument('--layer', type=int, default=3)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--sparse', type=bool, default=False)
    parser.add_argument('--no_boot', type=bool, default=False)
    parser.add_argument('--disable_global', type=bool, default=True)
    parser.add_argument('--with_update', type=bool, default=False)
    parser.add_argument('--is_multi', type=bool, default=True)
    parser.add_argument('--use_seed', type=int, default=1)
    parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
    args = parser.parse_args()
    return args


def print_settings(opt):
    print('################################')
    print('Dataset:\t\t', opt['dataset'])
    print('Saved Path:\t\t', opt['save'])
    print('Load Path:\t\t', opt['load'])
    print('Is multi-target:\t\t', opt['is_multi'])
    print('Bootstrapging Iter:\t', opt['iter'])
    print('Random seed:\t\t', opt['seed'])
    print('Dropout:\t\t', opt['dropout'])
    print('Learning Rate:\t\t', opt['lr'])
    print('Weight Decay:\t\t', opt['decay'])
    print('Candidate min match:\t', opt['min_match'])
    print('Unsupervised Scale:\t\t', opt['uns_scale'])
    print('Non-neighbor Scale:\t\t', opt['nn_scale'])
    print('Teacher pre Epoch:\t\t', opt['t_prepre_epoch'])
    print('Teacher Epoch:\t\t', opt['t_pre_epoch'])
    print('Student Epoch:\t\t', opt['s_pre_epoch'])
    print('Teacher draw:\t\t', opt['t_draw'])
    print('Student draw:\t\t', opt['s_draw'])
    print('use seed:\t\t', opt['use_seed'])
    print('Is sparse:\t\t', opt['sparse'])
    print('Layer of GCN:\t\t', opt['layer'])
    print('Use of Decoder:\t\t', not opt['no_boot'])
    print('Use of Global Att:\t', not opt['disable_global'])
    print('Initialization method:\t\t', opt['initialization'])
    print('################################')


def print_vocab(vocab):
    print('Tag\t---\tOriginal Tag')
    for i, tag in enumerate(vocab.itos):
        print(str(i) + '\t---\t'+str(tag))


def read_train(label_vocab, *train_files):
    seeds = {}
    for train_file in train_files:
        with open(train_file, 'r') as fi:
            for line in fi:
                node, lbl = line.strip().split('\t')
                ll = label_vocab.stoi[lbl]
                if ll not in seeds:
                    seeds[ll] = []
                seeds[ll].append(e_vocab.stoi[node])
    idx_train = []
    for ll, nodes in sorted(seeds.items(), key=lambda x: x[0]):
        idx_train.extend(nodes)
    return idx_train


def check_vocab(seeds, label_vocab, entity_vocab, n_per_class):
    start = 0
    i = 0
    while start < len(seeds):
        end = n_per_class + start
        for ii in seeds[start:end]:
            line = str(entity_vocab.itos[ii])
            line += ':\t' + str(label_vocab.itos[i])
            print(line)
        i += 1
        start = end


if __name__ == '__main__':
    args = regist_parser()
    if args.seed:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
    if args.cpu and args.device is not None or not torch.cuda.is_available():
        args.device = torch.device('cpu')
        args.s_device = torch.device('cpu')
    else:
        args.device = torch.device('cuda:%d' % args.device)
        args.s_device = torch.device('cuda:%d' % args.s_device)
        if args.seed:
            torch.cuda.manual_seed(args.seed)

    opt = vars(args)
    device = opt['device']
    s_device = opt['s_device']

    net_file = opt['dataset'] + '/net.txt'
    label_file = opt['dataset'] + '/label.txt'

    if opt['bert']:
        e_feature_file = opt['dataset'] + '/entity_bert_feature.txt'
        p_feature_file = opt['dataset'] + '/pattern_bert_feature.txt'
    else:
        e_feature_file = opt['dataset'] + '/entity_emb_feature.txt'
        p_feature_file = opt['dataset'] + '/pattern_emb_feature.txt'
    train_file = opt['dataset'] + '/seeds.txt'
    dev_file1 = opt['dataset'] + '/dev1.txt'
    dev_file2 = opt['dataset'] + '/dev2.txt'
    dev_file3 = opt['dataset'] + '/dev3.txt'
    opt['n_per_class'] = 10

    e_vocab = Vocab.from_file(net_file, [0])
    p_vocab = Vocab.from_file(net_file, [1])
    e_label_vocab = Vocab.from_file(label_file, [1])

    opt['num_node'] = len(e_vocab)
    opt['num_pattern'] = len(p_vocab)
    opt['num_class'] = len(e_label_vocab)
    opt['seed_count'] = 10
    opt['quick_save'] = opt['dataset']+'/quick_save'
    opt['save'] = os.path.join(opt['save'], os.path.basename(opt['dataset']))
    inner_dir = 'layer_%d_pre_%d' % (opt['layer'], opt['t_pre_epoch'])
    if opt['no_boot']:
        inner_dir += '_no_boot'
    if opt['disable_global']:
        inner_dir += '_no_global'
    if opt['initialization'] != 'embedding':
        inner_dir += '_' + opt['initialization']
    opt['save'] = os.path.join(opt['save'], inner_dir)
    if opt['gold']:
        opt['save'] = os.path.join(opt['save'], opt['gold'])

    graph = BootGraph.from_file(net_file, [e_vocab, p_vocab, 0, 1])
    label = EntityLabel.from_file(label_file, [e_vocab, 0], [e_label_vocab, 1])
    multi_label = MultiEntityLabel.from_file(
        label_file, [e_vocab, 0], [e_label_vocab, 1])

    if opt['initialization'] != 'embedding':
        if opt['initialization'] == 'random':
            es = torch.randn(len(e_vocab), 50, requires_grad=True)
            ps = torch.randn(len(p_vocab), 50, requires_grad=True)
        elif opt['initialization'] == 'uniform':
            es = torch.full((len(e_vocab), 50), 0.02, requires_grad=True)
            ps = torch.full((len(p_vocab), 50), 0.02, requires_grad=True)
        else:
            raise ValueError('initialization error %s' % opt['initialization'])
    else:
        if opt['bert']:
            feature_dim = 768
        else:
            feature_dim = 50
        es = Feature.from_file(
            e_feature_file, [e_vocab, 0], [feature_dim, 1], link=False)
        ps = Feature.from_file(
            p_feature_file, [p_vocab, 0], [feature_dim, 1], link=False)
        es.to_one_hot(normalize=False)
        ps.to_one_hot(normalize=False)
        es = torch.Tensor(es.one_hot)
        ps = torch.Tensor(ps.one_hot)

    opt['num_feature'] = es.size(1)
    if os.path.exists(opt['quick_save']+'/non_neighbor.pt'):
        adj = None
    else:
        adj = graph.get_adjacency(device, sparse=opt['sparse'])

    idx_train = read_train(e_label_vocab, train_file)
    idx_dev = read_train(e_label_vocab, dev_file1, dev_file2, dev_file3)
    idx_all = list(range(opt['num_node']))

    target = torch.LongTensor(label.itol)
    multi_target = multi_label.to_tensor(len(e_label_vocab.itos))
    idx_train = torch.LongTensor(idx_train)
    idx_dev = torch.LongTensor(idx_dev)
    idx_all = torch.LongTensor(idx_all)

    teacher = BootTeacher(opt)
    student = BootNet(opt)

    if device is not None:
        es, ps = es.to(device), ps.to(device)
        target = target.to(device)
        multi_target = multi_target.to(device)
        idx_train = idx_train.to(device)
        idx_dev = idx_dev.to(device)
        idx_all = idx_all.to(device)
        teacher = teacher.to(device)
        student = student.to(s_device)

    s_parameters = [p for p in student.parameters() if p.requires_grad]
    if s_parameters:
        s_optimizer = get_optimizer(opt['optimizer'], s_parameters,
                                    opt['lr'], opt['decay'])
    else:
        s_optimizer = None

    t_parameters = [p for p in teacher.parameters() if p.requires_grad]
    t_optimizer = get_optimizer(opt['optimizer'], t_parameters,
                                opt['lr'], opt['decay'])
    if opt['load'] and os.path.exists(opt['load']):
        t_optimizer = teacher.load(t_optimizer, opt['load'], device=device)

    if opt['s_load'] and s_parameters and \
            os.path.exists(opt['s_load']):
        s_optimizer = student.load(s_optimizer, opt['s_load'], device=s_device)

    print_settings(opt)
    print_vocab(e_label_vocab)

    if opt['no_boot']:
        noboot_train(opt, teacher, t_optimizer, student, s_optimizer,
                     target, multi_target, idx_train, es, ps, adj)
    else:
        boot_train(opt, teacher, t_optimizer, student, s_optimizer,
                   target, multi_target, idx_train, es, ps, adj, idx_dev=idx_dev, teacher_save=len(opt['gold']) > 0)
