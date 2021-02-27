"""
Evaluate model on dev and test set.

sample usage: 
python evaluate.py -m models/english.role.v0.3.mdl -d input/bert_large_processed_ace_notime/

The data path takes in the processed data in oneie format. 
It is expected to find train.oneie.json etc in the folder.
"""
import os
import json
import time
from argparse import ArgumentParser

import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import (BertConfig, AdamW,
                          get_linear_schedule_with_warmup)
from transformers import (BertTokenizer, RobertaTokenizer, XLMRobertaTokenizer,
                           AlbertTokenizer)
from tensorboardX import SummaryWriter

from src.model import OneIE
from src.graph import Graph
from src.config import Config
from src.data import IEDataset
from src.scorer import score_graphs
from src.util import generate_vocabs, load_valid_patterns, save_result, best_score_by_task
from predict import load_model

# configuration
parser = ArgumentParser()
parser.add_argument('-m', '--model_path', default='models/english.role.v0.3.mdl')
parser.add_argument('-d', '--data_path', default='input/bert_large_processed_ace_notime/')
parser.add_argument('-b', '--eval_batch_size', default=1, type=int)
parser.add_argument('--maxlen', default=128, type=int)
parser.add_argument('-g', '--gpu', default=False)
args = parser.parse_args()

# load model
model, tokenizer, config = load_model(args.model_path)

# load data path
data_path = args.data_path
train_file = data_path + 'train.oneie.json'
dev_file = data_path + 'dev.oneie.json'
test_file = data_path + 'test.oneie.json'

train_set = IEDataset(train_file, gpu=args.gpu,
                      relation_mask_self=config.relation_mask_self,
                      relation_directional=config.relation_directional,
                      symmetric_relations=config.symmetric_relations,
                      ignore_title=config.ignore_title,
                      max_length=args.maxlen)
dev_set = IEDataset(dev_file, gpu=args.gpu,
                    relation_mask_self=config.relation_mask_self,
                    relation_directional=config.relation_directional,
                    symmetric_relations=config.symmetric_relations,
                    max_length=args.maxlen)
test_set = IEDataset(test_file, gpu=args.gpu,
                     relation_mask_self=config.relation_mask_self,
                     relation_directional=config.relation_directional,
                     symmetric_relations=config.symmetric_relations, 
                     max_length=args.maxlen)
vocabs = model.vocabs

train_set.numberize(tokenizer, vocabs)
dev_set.numberize(tokenizer, vocabs)
test_set.numberize(tokenizer, vocabs)
valid_patterns = load_valid_patterns('resource/valid_patterns', vocabs)


# Set batch batch_size
config.eval_batch_size = args.eval_batch_size

batch_num = len(train_set) // config.batch_size
train_eval_batch_num = len(train_set) // config.eval_batch_size + \
    (len(train_set) % config.eval_batch_size != 0)
dev_batch_num = len(dev_set) // config.eval_batch_size + \
    (len(dev_set) % config.eval_batch_size != 0)
test_batch_num = len(test_set) // config.eval_batch_size + \
    (len(test_set) % config.eval_batch_size != 0)


# dev set
progress = tqdm.tqdm(total=dev_batch_num, ncols=75,
                     desc='Dev')
best_dev_role_model = False
dev_gold_graphs, dev_pred_graphs, dev_sent_ids, dev_tokens = [], [], [], []
for batch in DataLoader(dev_set, batch_size=config.eval_batch_size,
                        shuffle=False, collate_fn=dev_set.collate_fn):
    progress.update(1)
    graphs = model.predict(batch)
    if config.ignore_first_header:
        for inst_idx, sent_id in enumerate(batch.sent_ids):
            if int(sent_id.split('-')[-1]) < 4:
                graphs[inst_idx] = Graph.empty_graph(vocabs)
    for graph in graphs:
        graph.clean(relation_directional=config.relation_directional,
                    symmetric_relations=config.symmetric_relations)
    dev_gold_graphs.extend(batch.graphs)
    dev_pred_graphs.extend(graphs)
    dev_sent_ids.extend(batch.sent_ids)
    dev_tokens.extend(batch.tokens)
progress.close()
dev_scores = score_graphs(dev_gold_graphs, dev_pred_graphs,
                          relation_directional=config.relation_directional)


# test set
progress = tqdm.tqdm(total=test_batch_num, ncols=75,
                     desc='Test')
test_gold_graphs, test_pred_graphs, test_sent_ids, test_tokens = [], [], [], []
for batch in DataLoader(test_set, batch_size=config.eval_batch_size, shuffle=False,
                        collate_fn=test_set.collate_fn):
    progress.update(1)
    graphs = model.predict(batch)
    if config.ignore_first_header:
        for inst_idx, sent_id in enumerate(batch.sent_ids):
            if int(sent_id.split('-')[-1]) < 4:
                graphs[inst_idx] = Graph.empty_graph(vocabs)
    for graph in graphs:
        graph.clean(relation_directional=config.relation_directional,
                    symmetric_relations=config.symmetric_relations)
    test_gold_graphs.extend(batch.graphs)
    test_pred_graphs.extend(graphs)
    test_sent_ids.extend(batch.sent_ids)
    test_tokens.extend(batch.tokens)
progress.close()
test_scores = score_graphs(test_gold_graphs, test_pred_graphs,
                           relation_directional=config.relation_directional)
