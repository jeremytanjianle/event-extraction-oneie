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

# configuration
parser = ArgumentParser()
parser.add_argument('-c', '--config', default='config/example.json')
parser.add_argument("--train_file", default=None, type=str)
parser.add_argument("--dev_file", default=None, type=str)
parser.add_argument("--test_file", default=None, type=str)
parser.add_argument("--log_path", default=None, type=str)
parser.add_argument("--valid_pattern_path", default=None, type=str)
parser.add_argument("--bert_cache_dir", default=None, type=str)
parser.add_argument("--bert_model_name", default=None, type=str)
parser.add_argument("--global_features", default=None, nargs="*")
parser.add_argument("--symmetric_relations", default=None, nargs="*")
parser.add_argument("--sent_max_length", default=None, type=int)
parser.add_argument("--use_gpu", default=False, action='store_true')
args = parser.parse_args()

print(f'Config file: {args.config}')
config = Config.from_json_file(args.config)

# overwrite config args if manually set from the command line
if args.train_file is not None: config.train_file = args.train_file
if args.dev_file is not None: config.dev_file = args.dev_file
if args.test_file is not None: config.test_file = args.test_file
if args.log_path is not None: config.log_path = args.log_path
if args.valid_pattern_path is not None: config.valid_pattern_path = args.valid_pattern_path
if args.bert_cache_dir is not None: config.bert_cache_dir = args.bert_cache_dir
if args.bert_model_name is not None: config.bert_model_name = args.bert_model_name
if args.global_features is not None: config.global_features = args.global_features
if args.symmetric_relations is not None: config.symmetric_relations = args.symmetric_relations
if args.sent_max_length is not None: config.sent_max_length = args.sent_max_length
if args.use_gpu is not None: config.use_gpu = args.use_gpu


print(args.global_features)
print(args.symmetric_relations)
writer = SummaryWriter()

# set GPU device
use_gpu = config.use_gpu
if use_gpu and config.gpu_device >= 0:
    torch.cuda.set_device(config.gpu_device)

# output
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
output_dir = os.path.join(config.log_path, timestamp)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
log_file = os.path.join(output_dir, 'log.txt')
with open(log_file, 'w', encoding='utf-8') as w:
    w.write(json.dumps(config.to_dict()) + '\n')
    print('Log file: {}'.format(log_file))
best_role_model = os.path.join(output_dir, 'best.role.mdl')
dev_result_file = os.path.join(output_dir, 'result.dev.json')
test_result_file = os.path.join(output_dir, 'result.test.json')
last_model = os.path.join(output_dir, 'last.mdl')

# datasets
model_name = config.bert_model_name
# tokenizer = BertTokenizer.from_pretrained(model_name,
#                                           cache_dir=config.bert_cache_dir,
#                                           do_lower_case=False)
if config.bert_model_name.startswith('bert-'):
    tokenizer = BertTokenizer.from_pretrained(model_name, # cache_dir=config.bert_cache_dir,
                                              do_lower_case=False)
elif config.bert_model_name.startswith('roberta-'):
    tokenizer = RobertaTokenizer.from_pretrained(model_name, # cache_dir=config.bert_cache_dir,
                                            do_lower_case=False)
elif config.bert_model_name.startswith('xlm-roberta-'):
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_name, # cache_dir=config.bert_cache_dir,
                                            do_lower_case=False)
elif config.bert_model_name.startswith('albert-'):
    # "albert-xlarge-v2"
    tokenizer = AlbertTokenizer.from_pretrained(model_name, # cache_dir=config.bert_cache_dir,
                                                do_lower_case=False)
else:
    raise ValueError('Unknown model: {}'.format(config.bert_model_name))
train_set = IEDataset(config.train_file, gpu=use_gpu,
                      relation_mask_self=config.relation_mask_self,
                      relation_directional=config.relation_directional,
                      symmetric_relations=config.symmetric_relations,
                      ignore_title=config.ignore_title,
                      max_length=config.sent_max_length)
dev_set = IEDataset(config.dev_file, gpu=use_gpu,
                    relation_mask_self=config.relation_mask_self,
                    relation_directional=config.relation_directional,
                    symmetric_relations=config.symmetric_relations,
                     max_length=config.sent_max_length)
test_set = IEDataset(config.test_file, gpu=use_gpu,
                     relation_mask_self=config.relation_mask_self,
                     relation_directional=config.relation_directional,
                     symmetric_relations=config.symmetric_relations,
                     max_length=config.sent_max_length)
vocabs = generate_vocabs([train_set, dev_set, test_set])

train_set.numberize(tokenizer, vocabs)
dev_set.numberize(tokenizer, vocabs)
test_set.numberize(tokenizer, vocabs)
valid_patterns = load_valid_patterns(config.valid_pattern_path, vocabs)

batch_num = len(train_set) // config.batch_size
dev_batch_num = len(dev_set) // config.eval_batch_size + \
    (len(dev_set) % config.eval_batch_size != 0)
test_batch_num = len(test_set) // config.eval_batch_size + \
    (len(test_set) % config.eval_batch_size != 0)

# initialize the model
model = OneIE(config, vocabs, valid_patterns)
model.load_bert(model_name, cache_dir=config.bert_cache_dir)
if use_gpu:
    model.cuda(device=config.gpu_device)

# optimizer
transformer_name = config.bert_model_name.split('-')[0]
param_groups = [
    {
        'params': [p for n, p in model.named_parameters() if n.startswith(transformer_name)],
        'lr': config.bert_learning_rate, 'weight_decay': config.bert_weight_decay
    },
    {
        'params': [p for n, p in model.named_parameters() if not n.startswith(transformer_name)
                   and 'crf' not in n and 'global_feature' not in n],
        'lr': config.learning_rate, 'weight_decay': config.weight_decay
    },
    {
        'params': [p for n, p in model.named_parameters() if not n.startswith(transformer_name)
                   and ('crf' in n or 'global_feature' in n)],
        'lr': config.learning_rate, 'weight_decay': 0
    }
]
optimizer = AdamW(params=param_groups)
schedule = get_linear_schedule_with_warmup(optimizer,
                                           num_warmup_steps=batch_num * config.warmup_epoch,
                                           num_training_steps=batch_num * config.max_epoch)

# model state
state = dict(model=model.state_dict(),
             config=config.to_dict(),
             vocabs=vocabs,
             valid=valid_patterns)

global_step = 0
global_feature_max_step = int(config.global_warmup * batch_num) + 1
print('global feature max step:', global_feature_max_step)

tasks = ['entity', 'trigger', 'relation', 'role']
best_dev = {k: 0 for k in tasks}
for epoch in range(config.max_epoch):
    print('Epoch: {}'.format(epoch))

    # training set
    progress = tqdm.tqdm(total=batch_num, ncols=75,
                         desc='Train {}'.format(epoch))
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(DataLoader(
            train_set, batch_size=config.batch_size // config.accumulate_step,
            shuffle=True, drop_last=True, collate_fn=train_set.collate_fn)):

        loss = model(batch)
        loss = loss * (1 / config.accumulate_step)
        loss.backward()

        if (batch_idx + 1) % config.accumulate_step == 0:
            progress.update(1)
            global_step += 1
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_clipping)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad()
    progress.close()

    # dev set
    progress = tqdm.tqdm(total=dev_batch_num, ncols=75,
                         desc='Dev {}'.format(epoch))
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
    for task in tasks:
        writer.add_scalar(f'data/dev/{task}', dev_scores[task]['f'], epoch) 
        if dev_scores[task]['f'] > best_dev[task]:
            best_dev[task] = dev_scores[task]['f']
            if task == 'role':
                print('Saving best role model')
                torch.save(state, best_role_model)
                best_dev_role_model = True
                save_result(dev_result_file,
                            dev_gold_graphs, dev_pred_graphs, dev_sent_ids,
                            dev_tokens)

    # test set
    progress = tqdm.tqdm(total=test_batch_num, ncols=75,
                         desc='Test {}'.format(epoch))
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
    for task in tasks:
        writer.add_scalar(f'data/test/{task}', test_scores[task]['f'], epoch) 
    if best_dev_role_model:
        save_result(test_result_file, test_gold_graphs, test_pred_graphs,
                    test_sent_ids, test_tokens)

    result = json.dumps(
        {'epoch': epoch, 'dev': dev_scores, 'test': test_scores})
    with open(log_file, 'a', encoding='utf-8') as w:
        w.write(result + '\n')
    print('Log file', log_file)

torch.save(state, last_model)
writer.close()

best_score_by_task(log_file, 'role')
