# coding=utf-8
"""
@Description:
@Author: Lingyong Yan
@Date: 2019-07-02 04:25:41
@LastEditTime: 2019-08-15 04:17:21
@LastEditors: Lingyong Yan
@Python release: 3.7
@Notes:
"""
import json


def generate_ground_data(input_file, output_file):
    inputs = []
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            inputs.append(line)

    tag_dict = {}
    for (entity, tag, _) in inputs:
        if tag not in tag_dict:
            tag_dict[tag] = []
        tag_dict[tag].append(entity)
    with open(output_file, 'w') as f:
        json.dump(tag_dict, f, indent=2)


def emboot2bootgraph(root):
    entity_vocab = root + '/' + 'entity_vocabulary.emboot.filtered.txt'
    entity_label = root + '/' + 'entity_label_counts_emboot.filtered.txt'
    pattern_vocab = root + '/' + 'pattern_vocabulary_emboot.filtered.txt'
    link_file = root + '/' + 'training_data_with_labels_emboot.filtered.txt'
    seed_file = root + '/' + 'seedset.json'

    seeds = json.load(open(seed_file, 'r'))

    def load_entity_label(file_name):
        label_set, labels = set(), dict()
        entities = {}
        with open(file_name, 'r') as f:
            for line in f:
                key, value, num = line.strip().split('\t')
                num = int(num)
                label_set.add(value.strip())
                if key not in entities:
                    entities[key] = {}
                entities[key][value] = entities[key].get(value, 0) + num
        print(len(label_set))
        for i, label in enumerate(sorted(label_set)):
            labels[label] = str(i)
        return entities, labels

    entity_labels, labels = load_entity_label(entity_label)

    def load_vocab(file_name):
        vocab = []
        stoi = {}
        count = 0
        with open(file_name, 'r') as f:
            for line in f:
                items = line.strip().split('\t')
                key = items[0]
                assert key not in stoi
                stoi[key] = count
                vocab.append(items[0].strip())
                count += 1
        return vocab, stoi
    entities, entity_ids = load_vocab(entity_vocab)
    patterns, pattern_ids = load_vocab(pattern_vocab)

    def load_link(file_name, entity_ids, pattern_ids):
        links = {}
        with open(file_name, 'r') as f:
            for line in f:
                items = line.strip().split('\t')
                e = items[1]
                for p in items[2:]:
                    e_id, p_id = entity_ids[e], pattern_ids[p]
                    w = links.get((e_id, p_id), 0)
                    links[(e_id, p_id)] = w + 1
        return links

    links = load_link(link_file, entity_ids, pattern_ids)

    with open(root+'/' + 'label_map.txt', 'w') as f:
        for key, value in labels.items():
            f.write(key+'\t'+str(value)+'\n')

    for entity in entities:
        values = entity_labels[entity]
        sorted_values = sorted(values.items(), key=lambda x: x[1], reverse=True)
        entity_labels[entity] = [s[0] for s in sorted_values]

    with open(root + '/' + 'entity.txt', 'w') as f:
        for entity in entities:
            ls = entity_labels[entity]
            s = ' '.join(ls)
            f.write(entity + '\t' + s + '\n')

    with open(root + '/' + 'pattern.txt', 'w') as f:
        for p in patterns:
            f.write(p+'\n')

    with open(root + '/' + 'label.txt', 'w') as f:
        for entity in entities:
            ls = entity_labels[entity]
            ls = [labels[l] for l in ls]
            s = ' '.join(ls)
            f.write(str(entity_ids[entity]) + '\t' + s + '\n')

    with open(root + '/' + 'net.txt', 'w') as f:
        for (e, p), w in sorted(links.items(), key=lambda x: (x[0][0], x[0][1])):
            f.write(str(e) + '\t' + str(p) + '\t' + str(w)+'\n')

    with open(root + '/' + 'seeds.txt', 'w') as f:
        for key, values in seeds.items():
            key_id = labels[key]
            for value in values:
                f.write(str(entity_ids[value]) + '\t' + key_id+'\n')


if __name__ == '__main__':
    emboot2bootgraph('../data/OntoNotes')
    generate_ground_data('../data/OntoNotes/entity_label_counts_emboot.filtered.txt',
                         '../data/OntoNotes/ground_truth.json')
