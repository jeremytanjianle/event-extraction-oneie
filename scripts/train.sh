#!/bin/bash

oneie_dir=$(dirname $( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd ))

train_ID="R1"
bert_model_name="bert-large-cased"
dataset="ace"
sent_max_length=512
global_features=(
"role_role"
"event_role_num"
"role_entity"
"multi_role"
"relation_entity"
"relation_role_role"
"relation_relation"
"multi_event"
)
symmetric_relations=(
"PER-SOC"
)

cd "${oneie_dir}" && python3 train.py -c ./config/train.json \
--train_file "$(dirname ${oneie_dir})/data/oneie/${dataset}/collated-data/${bert_model_name}/json/train.oneie.json" \
--dev_file "$(dirname ${oneie_dir})/data/oneie/${dataset}/collated-data/${bert_model_name}/json/dev.oneie.json" \
--test_file "$(dirname ${oneie_dir})/data/oneie/${dataset}/collated-data/${bert_model_name}/json/test.oneie.json" \
--log_path "${oneie_dir}/logs/${train_ID}" \
--valid_pattern_path "${oneie_dir}/resource/valid_patterns_${dataset}" \
--bert_cache_dir "$(dirname ${oneie_dir})/data/oneie/bert" \
--bert_model_name "${bert_model_name}" \
--global_features ${global_features[@]} \
--symmetric_relations ${symmetric_relations[@]} \
--sent_max_length "${sent_max_length}" \
#--use_gpu