cd ../

poetry run python preprocessing/process_dygiepp.py -i ../data/rams/collated-data/default-settings/json/dev.json -o ../data/oneie/rams/collated-data/$1/json/dev.oneie.json -b $1

poetry run python preprocessing/process_dygiepp.py -i ../data/rams/collated-data/default-settings/json/test.json -o ../data/oneie/rams/collated-data/$1/json/test.oneie.json -b $1

poetry run python preprocessing/process_dygiepp.py -i ../data/rams/collated-data/default-settings/json/train.json -o ../data/oneie/rams/collated-data/$1/json/train.oneie.json -b $1

poetry run python preprocessing/check_zero_token_len.py -i data/oneie/acepp/collated-data/$1/json/dev.oneie.json

poetry run python preprocessing/check_zero_token_len.py -i data/oneie/acepp/collated-data/$1/json/test.oneie.json

poetry run python preprocessing/check_zero_token_len.py -i data/oneie/acepp/collated-data/$1/json/train.oneie.json

eai data push dsta.eventextraction.datasets@latest ../data/oneie/rams:oneie/rams

