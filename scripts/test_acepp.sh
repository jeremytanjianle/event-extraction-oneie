#!/bin/bash

oneie_dir=$(dirname $( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd ))

echo ${oneie_dir}
train_ID="T10"
bert_model_name="bert-large-uncased"
best_train_ID="T7"
best_model_date="20201109_035735"
dataset="acepp"
test_type="ace-event" # ace-event, rams
sent_max_length=512

cd "${oneie_dir}" && python3 predict.py \
--max_len "${sent_max_length}" \
-m "${oneie_dir}/logs/${best_train_ID}/${best_model_date}/best.role.mdl" \
-i "$(dirname ${oneie_dir})/data/oneie/${dataset}/collated-data/${bert_model_name}/test_json/${test_type}" \
-o "$(dirname ${oneie_dir})/data/oneie/${dataset}/collated-data/${bert_model_name}/test_json/${test_type}" \
--format json \
--gpu

#cd "${oneie_dir}" && python3 predict.py \
#--max_len "${sent_max_length}" \
#-m "${oneie_dir}/logs/${best_train_ID}/${best_model_date}/best.role.mdl" \
#-i "$(dirname ${oneie_dir})/data/oneie/cc" \
#-o "$(dirname ${oneie_dir})/oneie/logs/${train_ID}" \
#--format json \
#--gpu