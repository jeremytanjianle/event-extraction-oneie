import os
import json
import glob
import tqdm
import traceback
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from transformers import (BertTokenizer, BertConfig, RobertaTokenizer, XLMRobertaTokenizer,
                           AlbertTokenizer)

from src.model import OneIE
from src.config import Config
from src.util import save_result
from src.data import IEDatasetEval
from src.convert import json_to_cs

cur_dir = os.path.dirname(os.path.realpath(__file__))
format_ext_mapping = {'txt': 'txt', 'ltf': 'ltf.xml', 'json': 'json',
                      'json_single': 'json'}

def load_model(model_path, device=0, gpu=False, beam_size=5):
    print('Loading the model from {}'.format(model_path))
    map_location = 'cuda:{}'.format(device) if gpu else 'cpu'
    print(f'Map location: {map_location}')
    state = torch.load(model_path, map_location=map_location)

    config = state['config']
    if type(config) is dict:
        config = Config.from_dict(config)
    config.bert_cache_dir = os.path.join(cur_dir, 'bert')
    vocabs = state['vocabs']
    valid_patterns = state['valid']

    # recover the model
    model = OneIE(config, vocabs, valid_patterns)
    model.load_state_dict(state['model'])
    model.beam_size = beam_size

    if gpu:
        print('Setting cuda device...')
        model.cuda(device)

    # tokenizer = BertTokenizer.from_pretrained(config.bert_model_name,
    #                                           cache_dir=config.bert_cache_dir,
    #                                           do_lower_case=False)

    if config.bert_model_name.startswith('bert-'):
        tokenizer = BertTokenizer.from_pretrained(config.bert_model_name, cache_dir=config.bert_cache_dir,
                                                do_lower_case=False)
    elif config.bert_model_name.startswith('roberta-'):
        tokenizer = RobertaTokenizer.from_pretrained(config.bert_model_name, cache_dir=config.bert_cache_dir,
                                                do_lower_case=False)
    elif config.bert_model_name.startswith('xlm-roberta-'):
        tokenizer = XLMRobertaTokenizer.from_pretrained(config.bert_model_name, cache_dir=config.bert_cache_dir,
                                                do_lower_case=False)
    elif config.bert_model_name.startswith('albert-'):
        # "albert-xlarge-v2"
        tokenizer = AlbertTokenizer.from_pretrained(config.bert_model_name, cache_dir=config.bert_cache_dir,
                                                    do_lower_case=False)

    return model, tokenizer, config


def predict_document(path, model, tokenizer, config, batch_size=20, 
                     max_length=128, gpu=False, input_format='txt',
                     language='english'):
    """
    :param path (str): path to the input file.
    :param model (OneIE): pre-trained model object.
    :param tokenizer (BertTokenizer): BERT tokenizer.
    :param config (Config): configuration object.
    :param batch_size (int): Batch size (default=20).
    :param max_length (int): Max word piece number (default=128).
    :param gpu (bool): Use GPU or not (default=False).
    :param input_format (str): Input file format (txt or ltf, default='txt).
    :param langauge (str): Input document language (default='english').
    """
    test_set = IEDatasetEval(path, max_length=max_length, gpu=gpu,
                             input_format=input_format, language=language)
    test_set.numberize(tokenizer)
    # document info
    info = {
        'doc_id': test_set.doc_id,
        'ori_sent_num': test_set.ori_sent_num,
        'sent_num': len(test_set)
    }
    # prediction result
    result = []
    dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, 
                            collate_fn=test_set.collate_fn)
    progress = tqdm.tqdm(total=len(dataloader), ncols=75)
    for batch in dataloader:
        progress.update(1)
        graphs = model.predict(batch)
        for graph, tokens, sent_id, token_ids in zip(graphs, batch.tokens,
                                                     batch.sent_ids,
                                                     batch.token_ids):
            graph.clean(relation_directional=config.relation_directional,
                        symmetric_relations=config.symmetric_relations)
            result.append((sent_id, token_ids, tokens, graph))
    progress.close()
    return result, info


def predict(model_path, input_path, output_path, log_path=None, cs_path=None,
         batch_size=50, max_length=128, device=0, gpu=False,
         file_extension='txt', beam_size=5, input_format='txt',
         language='english'):
    """Perform information extraction.
    :param model_path (str): Path to the pre-trained model file.
    :param input_path (str): Path to the input directory.
    :param output_path (str): Path to the output directory.
    :param log_path (str): Path to the log file.
    :param cs_path (str): (optional) Path to the cold-start format output directory.
    :param batch_size (int): Batch size (default=50).
    :param max_length (int): Max word piece number for each sentence (default=128).
    :param device (int): GPU device index (default=0).
    :param gpu (bool): Use GPU (default=False).
    :param file_extension (str): Input file extension. Only files ending with the
    given extension will be processed (default='txt').
    :param beam_size (int): Beam size of the decoder (default=5).
    :param input_format (str): Input file format (txt or ltf, default='txt').
    :param language (str): Document language (default='english').
    """
    # set gpu device
    if gpu:
        torch.cuda.set_device(device)
    # load the model from file
    model, tokenizer, config = load_model(model_path, device=device, gpu=gpu,
                                          beam_size=beam_size)
    # get the list of documents
    file_list = glob.glob(os.path.join(input_path, '*.{}'.format(file_extension)))
    # log writer
    if log_path:
        log_writer = open(log_path, 'w', encoding='utf-8')
    # run the model; collect result and info
    doc_info_list = []
    # progress = tqdm.tqdm(total=len(file_list), ncols=75)
    for f in file_list:
        # progress.update(1)
        try:
            doc_result, doc_info = predict_document(
                f, model, tokenizer, config, batch_size=batch_size,
                max_length=max_length, gpu=gpu, input_format=input_format,
                language=language)
            # save json format result
            doc_id = doc_info['doc_id']
            with open(os.path.join(output_path, '{}.json'.format(doc_id)), 'w') as w:
                for sent_id, token_ids, tokens, graph in doc_result:
                    output = {
                        'doc_id': doc_id,
                        'sent_id': sent_id,
                        'token_ids': token_ids,
                        'tokens': tokens,
                        'graph': graph.to_dict()
                    }
                    w.write(json.dumps(output) + '\n')
            # write doc info
            if log_path:
                log_writer.write(json.dumps(doc_info) + '\n')
                log_writer.flush()
        except Exception as e:
            traceback.print_exc()
            if log_path:
                log_writer.write(json.dumps(
                    {'file': file, 'message': str(e)}) + '\n')
                log_writer.flush()
    # progress.close()

    # convert to the cold-start format
    if cs_path:
        print('Converting to cs format')
        json_to_cs(output_path, cs_path)

if __name__ == "__main__":
        
    parser = ArgumentParser()
    parser.add_argument('-m', '--model_path', help='path to the trained model')
    parser.add_argument('-i', '--input_dir', help='path to the input folder (ltf files)')
    parser.add_argument('-o', '--output_dir', help='path to the output folder (json files)')
    parser.add_argument('-l', '--log_path', default=None, help='path to the log file')
    parser.add_argument('-c', '--cs_dir', default=None, help='path to the output folder (cs files)')
    parser.add_argument('--gpu', action='store_true', help='use gpu')
    parser.add_argument('-d', '--device', default=0, type=int, help='gpu device index')
    parser.add_argument('-b', '--batch_size', default=10, type=int, help='batch size')
    parser.add_argument('--max_len', default=128, type=int, help='max sentence length')
    parser.add_argument('--beam_size', default=5, type=int, help='beam set size')
    parser.add_argument('--lang', default='english', help='Model language')
    parser.add_argument('--format', default='txt', help='Input format (txt, ltf, json)')

    args = parser.parse_args()
    extension = format_ext_mapping.get(args.format, 'ltf.xml')

    predict(
        model_path=args.model_path,
        input_path=args.input_dir,
        output_path=args.output_dir,
        cs_path=args.cs_dir,
        log_path=args.log_path,
        batch_size=args.batch_size,
        max_length=args.max_len,
        device=args.device,
        gpu=args.gpu,
        beam_size=args.beam_size,
        file_extension=extension,
        input_format=args.format,
        language=args.lang,
    )