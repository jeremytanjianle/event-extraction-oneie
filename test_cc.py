"""
python test_cc.py 
    --model_path models/T6t.mdl 
    --test_file inputs/cc/commoncrawl.json
    --log_path inputs/cc/T6t
"""

import os
import json
import time
from argparse import ArgumentParser

import torch
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer as Detok
from tqdm import tqdm

from src.data import IEDatasetEval, InstanceLdcEval, BatchLdcEval
from predict import load_model

nltk.download('punkt')

def text_to_tokens(text, sent_id, offset):
    """
    this tokenizes the text into words according to NLTK's model
    The important output is doc, which contains:
        (1) doc id
        (2) tokens
    """
    doc_tokens = []
    tokens = word_tokenize(text)
    tokens = [(token, offset + i, offset + i + 1)
              for i, token in enumerate(tokens)]
    doc_tokens.append((sent_id, tokens))
    return doc_tokens, tokens


def numberize(data, tokenizer, sent_max_length, sent_id):
    numberized_data = []
    for i, (sent_id, sent_tokens) in enumerate(data):
        tokens = []
        token_ids = []
        pieces = []
        token_lens = []
        truncate=False
        for token_text, start_char, end_char in sent_tokens:
            token_id = '{}:{}-{}'.format(sent_id, start_char, end_char)
            token_pieces = [p for p in tokenizer.tokenize(token_text) if p]
            if len(token_pieces) == 0:
                print("uncommon symbol encountered")
                print(sent_tokens)
                print()
                token_pieces = [p for p in tokenizer.tokenize('N') if p]
                # continue
            tokens.append(token_text)
            token_ids.append(token_id)
            # handle overlength sentences, by truncation
            if (len(token_pieces) + len(pieces))>(sent_max_length - 2):
                truncate=True
            if truncate:
                continue
            pieces.extend(token_pieces)
            token_lens.append(len(token_pieces))
            
        # # skip overlength sentences
        # if len(pieces) > sent_max_length - 2:
        #     continue
        # skip empty sentences
        if len(pieces) == 0:
            continue

        # pad word pieces with special tokens
        piece_idxs = tokenizer.encode(pieces,
                                      add_special_tokens=True,
                                      max_length=sent_max_length,
                                      truncation=True
                                      )
        pad_num = sent_max_length - len(piece_idxs)
        attn_mask = [1] * len(piece_idxs) + [0] * pad_num
        piece_idxs = piece_idxs + [0] * pad_num

        instance = InstanceLdcEval(
            sent_id=sent_id,
            tokens=tokens,
            token_ids=token_ids,
            pieces=pieces,
            piece_idxs=piece_idxs,
            token_lens=token_lens,
            attention_mask=attn_mask
        )
        numberized_data.append(instance)
    return numberized_data

def collate_fn(batch, use_gpu=False):
    batch_piece_idxs = []
    batch_tokens = []
    batch_token_lens = []
    batch_attention_masks = []
    batch_sent_ids = []
    batch_token_ids = []
    batch_token_nums = []

    for inst in batch:
        batch_piece_idxs.append(inst.piece_idxs)
        batch_attention_masks.append(inst.attention_mask)
        batch_token_lens.append(inst.token_lens)
        batch_tokens.append(inst.tokens)
        batch_sent_ids.append(inst.sent_id)
        batch_token_ids.append(inst.token_ids)
        batch_token_nums.append(len(inst.tokens))

    if use_gpu:
        batch_piece_idxs = torch.cuda.LongTensor(batch_piece_idxs)
        batch_attention_masks = torch.cuda.FloatTensor(
            batch_attention_masks)
        batch_token_nums = torch.cuda.LongTensor(batch_token_nums)
    else:
        batch_piece_idxs = torch.LongTensor(batch_piece_idxs)
        batch_attention_masks = torch.FloatTensor(
            batch_attention_masks)
        batch_token_nums = torch.LongTensor(batch_token_nums)

    return BatchLdcEval(sent_ids=batch_sent_ids,
                        token_ids=batch_token_ids,
                        tokens=batch_tokens,
                        piece_idxs=batch_piece_idxs,
                        token_lens=batch_token_lens,
                        attention_masks=batch_attention_masks,
                        token_nums=batch_token_nums)


def prepare_text(text, offset, tokenizer, sent_max_length, use_gpu, sent_id='asd'):
    data, tokens = text_to_tokens(text, sent_id, offset)
    data = numberize(data, tokenizer, sent_max_length, offset)
    data = collate_fn(data, use_gpu)
    return data, tokens


def get_graph_task_attribute(graph, task):
    if task == 'entities':
        return graph.entities, graph.entity_scores
    if task == 'triggers':
        return graph.triggers, graph.trigger_scores
    if task == 'relations':
        return graph.relations, graph.relation_scores
    if task == 'roles':
        return graph.roles, graph.role_scores
    return None, None


def get_task_type(task):
    if task == 'entities':
        return 'entity_type'
    if task == 'triggers':
        return 'event_type'
    if task == 'relations':
        return 'relation_type'
    if task == 'roles':
        return 'role_type'
    return None


def get_predictions_scores(graph):
    task_list = ['entities', 'triggers', 'relations', 'roles']
    output = {}

    for task in task_list:
        pred_list = []
        pred_task, pred_task_scores = get_graph_task_attribute(graph, task)
        for idx, entity in enumerate(pred_task):
            start, end, entity_type = entity
            itos = {i: s for s, i in graph.vocabs[get_task_type(task)].items()}
            label = itos[entity_type]
            pred_list.append([start, end, label, pred_task_scores[idx]])

        output[task] = pred_list

    return output


def predict(article, model, tokenizer, config, sent_max_length, use_gpu=False):
    detokenizer = Detok()
    sentences = article.get('sentences')

    sentence_list = []
    offset = 0
    for idx, text_list in enumerate(sentences):
        doc_id = f'{article.get("filename")}'
        sent_id = f'{article.get("filename")}-{idx}'
        text = detokenizer.detokenize(text_list)
        # text = ' '.join(text_list)
        data, tokens = prepare_text(text, offset, tokenizer, sent_max_length, use_gpu, sent_id)
        offset += len(tokens)
        graph = model.predict(data)
        graph = graph[0]
        graph.clean(relation_directional=config.relation_directional,
                    symmetric_relations=config.symmetric_relations)
        scores = get_predictions_scores(graph)
        # sentence_pred = {
        #     'sent_id': sent_id,
        #     'pred': scores,
        #     'tokens': text_list
        # }
        sentence_pred = {
            'doc_id':doc_id, 
            'sent_id': sent_id,
            'token_ids': data.token_ids[0], # this works because script takes in single sentences only
            'tokens': [token[0] for token in tokens], # text_list,
            'graph': graph.to_dict()
        }
        sentence_list.append(sentence_pred)

    return sentence_list

def count_maxlen_of_articles(articles):
    """
    This will estimate highest nunmber of tokens in the sentences in articles
    Allowing us to set an appropriate token length for our transformer
    which is O(n^3) in complexity to sentence len n
    """
    def count_sentence_len(doc):
        sentences = doc['sentences']
        return max([len(sentence) for sentence in sentence])

    return max([count_sentence_len(doc) for doc in articles])

if __name__ == "__main__":

    # configuration
    parser = ArgumentParser()
    parser.add_argument('--model_path', default=None, type=str)
    parser.add_argument("--test_file", default=None, type=str)
    parser.add_argument("--log_path", default=None, type=str)
    parser.add_argument("--sent_max_length", default=512, type=int)
    parser.add_argument("--use_gpu", default=False, action='store_true')
    args = parser.parse_args()

    # output
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    output_dir = os.path.join(args.log_path, timestamp)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    predictions_file = os.path.join(output_dir, f'predictions.json')

    log_file = os.path.join(output_dir, 'log.txt')
    with open(log_file, 'w', encoding='utf-8') as w:
        print('Log file: {}'.format(log_file))

    print(f'Will use GPU: {args.use_gpu}')

    # load the model from file
    model, tokenizer, config = load_model(args.model_path,
                                        device=0,
                                        gpu=args.use_gpu,
                                        beam_size=5)

    # set GPU device
    if args.use_gpu:
        torch.cuda.set_device(0)

    # load cc json
    with open(args.test_file, 'r') as f, open(predictions_file, 'a') as output_f:
        articles = [json.loads(a) for a in f.readlines()]
        print(f'Loaded {len(articles)} articles from {args.test_file}.')
        print(f'Found token max length of {count_maxlen_of_articles(articles)} in {args.test_file}')

        # iterate through each article
        progress = tqdm(total=len(articles), ncols=75)
        for idx, article in enumerate(articles):
            progress.update(1)
            predicted_article_events = predict(article, model, tokenizer, config, args.sent_max_length)

            for sent in predicted_article_events:
                # save the articles with predictions back into a json
                json.dump(sent, output_f)
                output_f.write('\n')

    print('Done.')
