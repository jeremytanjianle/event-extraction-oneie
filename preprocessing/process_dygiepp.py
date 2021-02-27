import json
from argparse import ArgumentParser

from numpy.core import unicode
from transformers import (BertTokenizer,
                          RobertaTokenizer,
                          XLMRobertaTokenizer,
                          PreTrainedTokenizer,
                          AlbertTokenizer)
from tqdm import tqdm
import os
import re

def map_index(pieces):
    idxs = []
    for i, piece in enumerate(pieces):
        if i == 0:
            idxs.append([0, len(piece)])
        else:
            _, last = idxs[-1]
            idxs.append([last, last + len(piece)])
    return idxs


def convert(input_file, output_file, tokenizer, num_event_layers):
    # creates output file if it does not exists
    output_folder = '/'.join(output_file.split('/')[0:-1])
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(input_file, 'r', encoding='utf-8') as r, \
            open(output_file, 'w', encoding='utf-8') as w:
        for line in tqdm(r):
            doc = json.loads(line)
            doc_id = doc['doc_key']
            sentences = doc['sentences']
            sent_num = len(sentences)
            entities = doc.get('ner', [[] for _ in range(sent_num)])
            relations = doc.get('relations', [[] for _ in range(sent_num)])
            events = doc.get('events', [[] for _ in range(sent_num)])

            offset = 0
            for i, (sent_tokens, sent_entities, sent_relations, sent_events) in enumerate(zip(
                sentences, entities, relations, events
            )):

                sent_id = '{}-{}'.format(doc_id, i)
                pieces = [tokenizer.tokenize(t) for t in sent_tokens]

                # replaces pieces that are not tokenized (ie. return [] with "[UNK]"
                pieces = [piece if piece != [] else ['[UNK]'] for piece in pieces]

                word_lens = [len(p) for p in pieces]
                idx_mapping = map_index(pieces)

                sent_entities_ = []
                sent_entity_map = {}
                for j, (start, end, entity_type) in enumerate(sent_entities):
                    start, end = start - offset, end - offset + 1
                    entity_id = '{}-E{}'.format(sent_id, j)
                    entity = {
                        'id': entity_id,
                        'start': start, 'end': end,
                        'entity_type': entity_type,
                        # Mention types are not included in DyGIE++'s format
                        'mention_type': 'UNK',
                        'text': ' '.join(sent_tokens[start:end])}
                    sent_entities_.append(entity)
                    sent_entity_map[start] = entity

                sent_relations_ = []
                for j, (start1, end1, start2, end2, rel_type) in enumerate(sent_relations):
                    start1, end1 = start1 - offset, end1 - offset
                    start2, end2 = start2 - offset, end2 - offset
                    arg1 = sent_entity_map[start1]
                    arg2 = sent_entity_map[start2]
                    relation_id = '{}-R{}'.format(sent_id, j)
                    rel_type = rel_type.split('.')[0]
                    relation = {
                        'relation_type': rel_type,
                        'id': relation_id,
                        'arguments': [
                            {
                                'entity_id': arg1['id'],
                                'text': arg1['text'],
                                'role': 'Arg-1'
                            },
                            {
                                'entity_id': arg2['id'],
                                'text': arg2['text'],
                                'role': 'Arg-2'
                            },
                        ]
                    }
                    sent_relations_.append(relation)

                sent_events_ = []
                for j, event in enumerate(sent_events):
                    event_id = '{}-EV{}'.format(sent_id, j)
                    if len(event[0]) == 3:
                        trigger_start, trigger_end, event_type = event[0]
                    elif len(event[0]) == 2:
                        trigger_start, event_type = event[0]
                        trigger_end = trigger_start
                    trigger_start, trigger_end = trigger_start - offset, trigger_end - offset + 1
                    event_type = event_type.replace('.', ':').replace('-', '')

                    # reduce number of layers in event_type
                    event_type_layer_list = event_type.split(':')
                    if len(event_type_layer_list) < num_event_layers:
                        num_event_layers = len(event_type_layer_list)
                    event_type = ':'.join(event_type_layer_list[0:num_event_layers])

                    args = event[1:]
                    args_ = []
                    for arg_start, arg_end, role in args:
                        arg_start, arg_end = arg_start - offset, arg_end - offset
                        arg = sent_entity_map[arg_start]
                        args_.append({
                            'entity_id': arg['id'],
                            'text': arg['text'],
                            'role': role
                        })
                    event_obj = {
                        'event_type': event_type,
                        'id': event_id,
                        'trigger': {
                            'start': trigger_start,
                            'end': trigger_end,
                            'text': ' '.join(sent_tokens[trigger_start:trigger_end])
                        },
                        'arguments': args_
                    }
                    sent_events_.append(event_obj)

                sent_ = {
                    'doc_id': doc_id,
                    'sent_id': sent_id,
                    'entity_mentions': sent_entities_,
                    'relation_mentions': sent_relations_,
                    'event_mentions': sent_events_,
                    'tokens': sent_tokens,
                    'pieces': [p for w in pieces for p in w],
                    'token_lens': word_lens,
                    'sentence': ' '.join(sent_tokens)
                }
                w.write(json.dumps(sent_) + '\n')

                offset += len(sent_tokens)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', help='Path to the input file')
    parser.add_argument('-o', '--output', help='Path to the output file')
    parser.add_argument('-b', '--bert', help='BERT model name', default='bert-large-cased')
    parser.add_argument('--num_event_layers', help='Number of event layers to keep for RAMS', default=3, choices=[1, 2, 3], type=int)
    args = parser.parse_args()


    # Create a tokenizer based on the model name
    model_name = args.bert
    if model_name.startswith('bert-'):
        tokenizer = BertTokenizer.from_pretrained(model_name,
                                                  do_lower_case=False)
    elif model_name.startswith('roberta-'):
        tokenizer = RobertaTokenizer.from_pretrained(model_name,
                                                     do_lower_case=False)
    elif model_name.startswith('xlm-roberta-'):
        tokenizer = XLMRobertaTokenizer.from_pretrained(model_name,
                                                        do_lower_case=False)
    elif model_name.startswith('albert-'):
        # "albert-xlarge-v2"
        tokenizer = AlbertTokenizer.from_pretrained(model_name,
                                                    do_lower_case=False)

    else:
        raise ValueError('Unknown model name: {}'.format(model_name))

    convert(args.input, args.output, tokenizer, args.num_event_layers)
