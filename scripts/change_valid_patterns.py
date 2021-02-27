"""
Update valid patterns in mdl file

python scripts/change_valid_patterns.py 
    -i models/T5.mdl 
    -o models/T5t.mdl 
    -v resource/valid_patterns_acepp
"""
import os
import sys
sys.path.append('.')
import json
import glob
import tqdm
import traceback
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from transformers import (BertTokenizer, BertConfig, RobertaTokenizer, XLMRobertaTokenizer,
                           AlbertTokenizer)

from model import OneIE
from config import Config
from util import save_result
from data import IEDatasetEval
from convert import json_to_cs
from predict import load_model
from util import generate_vocabs, load_valid_patterns, save_result, best_score_by_task

# configuration
parser = ArgumentParser()
parser.add_argument('-i', '--input', default='models/english.role.v0.3.mdl')
parser.add_argument('-o', '--output', default='models/english.role.v0.3.mdl')
parser.add_argument('-v', '--valid_patterns', default="resource/valid_patterns_acepp")
args = parser.parse_args()


old_model = args.input # 'models/T5.mdl'
new_model = args.output # 'models/T5t.mdl'
validpatternspath = args.valid_patterns # "resource/valid_patterns_acepp"

# load model and new valid patterns
model, tokenizer, config = load_model(old_model)
valid_patterns = load_valid_patterns(validpatternspath, model.vocabs)

# save updated model in output
state = dict(model=model.state_dict(),
             config=config.to_dict(),
             vocabs=model.vocabs,
             valid=valid_patterns)
torch.save(state, args.output)