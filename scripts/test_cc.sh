#!/bin/bash

oneie_dir=$(dirname $( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd ))

echo ${oneie_dir}
train_ID="C2"
best_train_ID="S2"
best_model_date="20201109_062654"
sent_max_length=512


cd "${oneie_dir}" && python3 test_cc.py -c ./config/train.json \
--test_file "$(dirname ${oneie_dir})/data/eeqa/commoncrawl.json" \
--log_path "${oneie_dir}/logs/${train_ID}" \
--model_path "${oneie_dir}/logs/${best_train_ID}/${best_model_date}/best.role.mdl" \
--sent_max_length "${sent_max_length}" \
--use_gpu
