cd ../

poetry run python preprocessing/process_ace.py -i ../data/ace_2005_dataset/data -o ../data/oneie/ace/collated-data/$1/json -s splits/ACE05-E -l english -b $1 --time_and_val 

eai data push dsta.eventextraction.datasets@latest ../data/oneie/ace:oneie/ace

