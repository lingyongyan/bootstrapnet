# coding=UTF-8
"""
@Description:
@Author: Lingyong Yan
@Date: 2019-07-04 09:28:11
@LastEditTime: 2019-08-15 01:10:37
@LastEditors: Lingyong Yan
@Python release: 3.7
@Notes:
"""
import json
import regex as re
from collections import OrderedDict

pat = re.compile(r'Epoch (\d+)')
nums = []
nums.extend(list(range(10, 251, 10)))


def read_result_lp(lp_file):
    outputs = OrderedDict()
    results = []
    it = 0
    with open(lp_file, 'r') as f:
        temp = dict()
        for line in f:
            line = line.strip()
            if pat.match(line):
                it = int(pat.match(line)[1])
                if it > 0 and temp:
                    results.append(temp)
                temp = dict()
                continue
            if it > -1:
                values = line.split('\t')
                label, words = values[0], values[1:]
                if label not in outputs:
                    outputs[label] = []
                if label not in temp:
                    temp[label] = []
                outputs[label].extend(words)
                temp[label].extend(words)
        if temp:
            results.append(temp)
    return results, outputs


def read_result_emb(lp_file):
    outputs = OrderedDict()
    with open(lp_file, 'r') as f:
        for line in f:
            values = line.strip().split('\t')
            word, label = values[1], values[2]
            if label not in outputs:
                outputs[label] = []
            outputs[label].append(word)
    return outputs


def eval_lp2(golden_dict, extracted_dict, logger=None):
    labels = list(golden_dict.keys())
    p_ns = {}
    for label, words in extracted_dict.items():
        precious, n = 0, 0
        golden = set(golden_dict[label])
        precious_n = []
        for word in words:
            if word == '</s>':
                continue
            n += 1
            if word in golden:
                precious += 1
            if n % 10 == 0:
                p = precious / n
                precious_n.append((n, p))
            if n > 250:
                break
        p_ns[label] = precious_n
    outputs = [['metrics'] + labels + ['AVE']]
    for i, num in enumerate(nums):
        current_output = ['P@%d' % num]
        ps = []
        for label in labels:
            if len(p_ns[label]) > i:
                p = p_ns[label][i][1]
                ps.append(p)
                current_output.append('%.3f' % p)
            else:
                current_output.append('')
        if len(ps) == len(labels):
            current_output.append('%.3f' % (sum(ps) / len(ps)))
        else:
            current_output.append('')
        outputs.append(current_output)
    for line in pre_func(outputs):
        if logger:
            logger.info(line)
        else:
            print(line)


def eval_lp(golden_dict, extracted_dict_list, logger=None):
    keys = list(golden_dict.keys())
    if logger:
        logger.info('\t'.join(keys))
    else:
        print('\t'.join(keys))

    total, precious = 0, 0
    for i, extracted_dict in enumerate(extracted_dict_list):
        line = ''
        for key in keys:
            words = extracted_dict[key] if key in extracted_dict else []
            p, n = 0, 0
            golden = set(golden_dict[key])
            for word in words:
                if word == '</s>':
                    continue
                n += 1
                if word in golden:
                    p += 1
            acc = float(p) / n if n else 0.
            line += '%d/%d(%.3f)\t' % (p, n, acc)
            total += n
            precious += p
        step_acc = float(precious) / total if total else 0.
        line += '%d\t%d\t%.4f' % (precious, total, step_acc)
        if logger:
            logger.info(line)
        else:
            print(line)


def pre_func(datas):
    for data in datas:
        yield '\t'.join([str(d) for d in data])


def judge(golden_set, extracted_set):
    precious_n = []
    precious, n = 0, 0
    for node in extracted_set:
        if node == '</s>':
            continue
        n += 1
        if node in golden_set:
            precious += 1
        if n % 10 == 0:
            p = precious / n
            precious_n.append((p, n))
        if n > 250:
            break
    return precious_n


if __name__ == '__main__':
    with open('data/OntoNotes/ground_truth.json', 'r') as f:
        golden_dict = json.load(f)
    # path = 'results/CoNLL/pools_output.txt'
    # path = 'results/CoNLL/lp_results.txt'
    # path = 'results/CoNLL/gupta_out.txt'
    path = 'results/OntoNotes/lp_results.txt'
    results, outputs = read_result_lp(path)
    eval_lp(golden_dict, results)
    eval_lp2(golden_dict, outputs)
