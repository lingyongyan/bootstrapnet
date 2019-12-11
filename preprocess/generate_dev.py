# coding=utf-8
"""
@Description:
@Author: Lingyong Yan
@Date: 2019-08-28 13:10:51
@LastEditTime: 2019-08-28 13:10:51
@LastEditors: Lingyong Yan
@Python release: 3.7
@Notes:
"""
import os
import numpy as np

_dir = '../data/OntoNotes/'


def read_labels(path):
    labels = {}
    with open(os.path.join(_dir, path), 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            if line:
                entity, label = line[0], line[1].split(' ')[0]
                if label not in labels:
                    labels[label] = set()
                labels[label].add(entity)
    return labels


def write_labels(path, data, start, end):
    with open(os.path.join(_dir, path), 'w') as f:
        for k, v in data.items():
            for _v in v[start:end]:
                f.write(str(_v)+'\t'+str(k)+'\n')

labels = read_labels('label.txt')
seeds = read_labels('seeds.txt')
dev = {}
total_dev = set()
for k, v in labels.items():
    print(k, ':', len(v))
    remain = list(v - seeds[k] - total_dev)
    if len(remain) > 10:
        if len(remain) >= 230:
            sample_num = 30
        elif len(remain) >= 220:
            sample_num = 20
        else:
            sample_num = 10
        print(k, 'remain', len(remain), 'sampled for', sample_num)
        sampled = list(np.random.choice(remain, sample_num, replace=False))
        total_dev.update(sampled)
        dev[k] = sampled
print(len(total_dev))

write_labels('dev1.txt', dev, 0, 10)
write_labels('dev2.txt', dev, 10, 20)
write_labels('dev3.txt', dev, 20, 30)
