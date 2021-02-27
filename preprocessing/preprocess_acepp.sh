cd ../

poetry run python preprocessing/process_dygiepp.py -i ../data/ace_event_plus_rams/collated-data/default-settings/json/dev.json -o ../data/oneie/acepp/collated-data/$1/json/dev.oneie.json -b $1
poetry run python preprocessing/process_dygiepp.py -i ../data/ace_event_plus_rams/collated-data/default-settings/json/test.json -o ../data/oneie/acepp/collated-data/$1/json/test.oneie.json -b $1
poetry run python preprocessing/process_dygiepp.py -i ../data/ace_event_plus_rams/collated-data/default-settings/json/test-rams.json -o ../data/oneie/acepp/collated-data/$1/json/test-rams.oneie.json -b $1
poetry run python preprocessing/process_dygiepp.py -i ../data/ace_event_plus_rams/collated-data/default-settings/json/test-ace-event.json -o ../data/oneie/acepp/collated-data/$1/json/test-ace-event.oneie.json -b $1
poetry run python preprocessing/process_dygiepp.py -i ../data/ace_event_plus_rams/collated-data/default-settings/json/train.json -o ../data/oneie/acepp/collated-data/$1/json/train.oneie.json -b $1

# Checking for zero token len
poetry run python preprocessing/check_zero_token_len.py -i data/oneie/acepp/collated-data/$1/json/dev.oneie.json
poetry run python preprocessing/check_zero_token_len.py -i data/oneie/acepp/collated-data/$1/json/test.oneie.json
poetry run python preprocessing/check_zero_token_len.py -i data/oneie/acepp/collated-data/$1/json/test-rams.oneie.json
poetry run python preprocessing/check_zero_token_len.py -i data/oneie/acepp/collated-data/$1/json/test-ace-event.oneie.json
poetry run python preprocessing/check_zero_token_len.py -i data/oneie/acepp/collated-data/$1/json/train.oneie.json


# Extract conflict only test data
poetry run python preprocessing/get_conflict_events.py -i data/oneie/acepp/collated-data/$1/json/test.oneie.json
poetry run python preprocessing/get_conflict_events.py -i data/oneie/acepp/collated-data/$1/json/test-rams.oneie.json
poetry run python preprocessing/get_conflict_events.py -i data/oneie/acepp/collated-data/$1/json/test-ace-event.oneie.json

eai data push dsta.eventextraction.datasets@latest ../data/oneie/acepp:oneie/acepp

