# coding=UTF-8
"""
@Description:
@Author: Lingyong Yan
@Date: 2019-07-03 05:39:58
@LastEditTime: 2019-08-15 01:10:34
@LastEditors: Lingyong Yan
@Python release: 3.7
@Notes:
"""
import os
import torch
from collections import OrderedDict

from torchtext.vocab import Vectors
from torchtext.data import Iterator
from core.encoder import CBOWEncoder
from core.util import PatternField, PatternDataset

from transformers import BertModel, BertTokenizer

from tqdm import tqdm


CBOW_PARAMS = '../caches/cbow_w2v.pkl'


def link_feature(root):
    net_file = os.path.join(root, 'net.txt')
    entity_feature_file = os.path.join(root, 'entity_feature.txt')
    pattern_feature_file = os.path.join(root, 'pattern_feature.txt')

    entity_features = OrderedDict()
    pattern_features = OrderedDict()
    with open(net_file, 'r') as f:
        for line in f:
            e, p, w = line.strip().split('\t')
            e, p, w = int(e), int(p), float(w)
            if e not in entity_features:
                entity_features[e] = []
            if p not in pattern_features:
                pattern_features[p] = []
            entity_features[e].append((p, w))
            pattern_features[p].append((e, w))

    with open(entity_feature_file, 'w') as f:
        for e, values in entity_features.items():
            features = [str(p)+':'+str(w) for p, w in values]
            f.write(str(e)+'\t'+' '.join(features)+'\n')

    with open(pattern_feature_file, 'w') as f:
        for e, values in sorted(pattern_features.items(), key=lambda x: x[0]):
            values = sorted(values, key=lambda x: x[0])
            features = [str(p)+':'+str(w) for p, w in values]
            f.write(str(e)+'\t'+' '.join(features)+'\n')


def embed(encoder, dataset_cls, sentences, field, batch_size=64):
    with torch.no_grad():
        dataset = dataset_cls(sentences, fields=[('data', field)])
        data_iter = Iterator(dataset,
                             batch_size=batch_size,
                             sort=False,
                             shuffle=False,
                             repeat=False)
        sentences_embeddings = []
        for data in iter(data_iter):
            w_ids, lengths = getattr(data, 'data')
            weighted_embeddings = encoder(w_ids, lengths)
            sentences_embeddings.append(weighted_embeddings)
    return torch.cat(sentences_embeddings, dim=0)


def embedding_feature(root):
    entity = os.path.join(root, 'entity.txt')
    pattern = os.path.join(root, 'pattern.txt')
    entity_feature_file = os.path.join(root, 'entity_emb_feature.txt')
    pattern_feature_file = os.path.join(root, 'pattern_emb_feature.txt')

    encoder, dataset_cls, field = load_encoder()
    entities = []
    patterns = []
    with open(entity, 'r') as f:
        for line in f:
            line = line.strip().split('\t')[0]
            entities.append(line)
    with open(pattern, 'r') as f:
        for line in f:
            line = line.strip().split('\t')[0]
            patterns.append(line)

    es = embed(encoder, dataset_cls, entities, field)
    ps = embed(encoder, dataset_cls, patterns, field)
    es = es.tolist()
    ps = ps.tolist()
    with open(entity_feature_file, 'w') as f:
        for e, embedding in enumerate(es):
            features = [str(emb) for emb in embedding]
            f.write(str(e) + '\t' + ' '.join(features) + '\n')

    with open(pattern_feature_file, 'w') as f:
        for p, embedding in enumerate(ps):
            features = [str(emb) for emb in embedding]
            f.write(str(p) + '\t' + ' '.join(features) + '\n')


def bert(bert_model, bert_tokenizer, sentences):
    embs = []
    for sentence in tqdm(sentences):
        input_ids = torch.tensor([bert_tokenizer.encode(sentence)])
        with torch.no_grad():
            bert_emb = bert_model(input_ids)[1]
            embs.append(bert_emb)
    return torch.cat(embs, dim=0)


def bert_feature(root):
    entity = os.path.join(root, 'entity.txt')
    pattern = os.path.join(root, 'pattern.txt')
    entity_feature_file = os.path.join(root, 'entity_bert_feature.txt')
    pattern_feature_file = os.path.join(root, 'pattern_bert_feature.txt')

    entities = []
    patterns = []
    with open(entity, 'r') as f:
        for line in f:
            line = line.strip().split('\t')[0]
            entities.append(line)
    with open(pattern, 'r') as f:
        for line in f:
            line = line.strip().split('\t')[0]
            patterns.append(line)

    bert_model = BertModel.from_pretrained('bert-base-uncased')
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    es = bert(bert_model, bert_tokenizer, entities)
    ps = bert(bert_model, bert_tokenizer, patterns)
    es = es.tolist()
    ps = ps.tolist()
    with open(entity_feature_file, 'w') as f:
        for e, embedding in enumerate(es):
            features = [str(emb) for emb in embedding]
            f.write(str(e) + '\t' + ' '.join(features) + '\n')

    with open(pattern_feature_file, 'w') as f:
        for p, embedding in enumerate(ps):
            features = [str(emb) for emb in embedding]
            f.write(str(p) + '\t' + ' '.join(features) + '\n')


def load_encoder():
    vocab_size = 400002
    dim = 50
    vec_path = 'caches'
    name = 'glove.6B.50d.txt'
    itos_path = 'caches/glove.6B.50d.vocab.pt'
    field = PatternField()
    dataset_cls = PatternDataset

    print('initialize CBOW encoder')
    if os.path.exists(itos_path):
        print('load itos from cache file %s' % itos_path)
        itos, stoi = torch.load(itos_path)
        field.build_vocab(itos=itos)
        encoder = CBOWEncoder(vocab_size=vocab_size, emb_dim=dim)
        print('load CBOW encoder parameters from %s' % CBOW_PARAMS)
        encoder.load_state_dict(torch.load(CBOW_PARAMS))
    else:
        vectors = Vectors(name, vec_path)
        torch.save((vectors.itos, vectors.stoi), itos_path)
        field.build_vocab(word_vectors=vectors)
        encoder = CBOWEncoder(vocab_size=vocab_size,
                              vectors=field.vocab.vectors)
        print('save CBOW encoder parameters to%s' % CBOW_PARAMS)
        torch.save(encoder.state_dict(), CBOW_PARAMS)
    return encoder, dataset_cls, field


if __name__ == '__main__':
    root = '../data/CoNLL'
    embedding_feature(root)
    # bert_feature(root)
