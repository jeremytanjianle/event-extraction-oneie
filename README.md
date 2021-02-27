OneIE v0.4.5

# Requirements

Python 3.5+
Python packages
- PyTorch 1.0+ (Install the CPU version if you use this tool on a machine without GPUs)
- transformers
- tqdm
- lxml
- nltk

This project uses Poetry to manage library versions.

1. Get the latest versions of the dependencies and to update the ``poetry.lock`` file.

```
poetry update
```

2. Install libraries.
```
poetry install
```

3. Spawns a shell within the virtual environment.
```
poetry shell
```

4. Ensure that the required libraries listed in ``pyproject.toml`` are installed in the virtual environment.
```
pip list
```

5. You're good to go! :)

## Creating venv for Jupyter notebooks

1. Ensure that both ``jupyter`` and ``ipykernel`` are in the dependencies.
```
poetry add -D jupyter ipykernel
```

2. Create virtual environments with ``ipython``.
```
poetry run ipython kernel install --user --name=oneie-venv
```

3. Start up Jupyter in the virtual environment.
```
poetry run jupyter notebook
```

4. In the notebook, **Kernel** > **Change Kernel** > **oneie-venv**. Restart the kernel. 


# How to Run

## Pre-processing

### Downloading RAMS
Leverage on existing parsed RAMS to DyGIE++ in toolkit:
```
eai data pull dsta.eventextraction.datasets@latest data/
``` 

### Preprocess RAMS data
ONEIE requires the data to be in its own format. We may transform data in the DyGIE++ format to ONEIE's format with: 
```
python oneie/preprocessing/process_dygiepp.py -i data/rams/collated-data/default-settings/json/dev.json -o data/oneie/rams/collated-data/default-settings/json/dev.json
```

Arguments:
- -i, --input: Path to the input file.
- -o, --output: Path to the output file.
- -b, --bert: Name of BERT model used for tokenization (default: bert-large-cased).

A sample DyGIE++ format:
```
{"doc_key":"nw_RC00e90a0209cf7c63e3faf5008f034002ef61cea93a159a31aa33e18e","sentences":[["Three","specific","points","illustrate","why","Americans","see","Trump","as","the","problem",":","1",")","Trump","has","trouble","working","with","people","beyond","his","base",".","In","Saddam","Hussein","'s","Iraq","that","might","work","when","opponents","can","be","thrown","in","jail","or","exterminated",".","In","the","United","States","that","wo","n't","fly",":","presidents","must","build","bridges","within","and","beyond","their","core","support","to","resolve","challenges",".","Without","alliances",",","a","president","ca","n't","get","approval","to","get","things","done","."]],"events":[[[[40,"life.die.n\/a"],[33,33,"victim"],[28,28,"place"]]]],"ner":[[[33,33,"victim"],[28,28,"place"]]],"relations":[[]],"_sentence_start":[0],"dataset":"rams"}
```

A sample ONEIE format after using `preprocessing/process_dygiepp.py`:
```
{"doc_id": "nw_RC00e90a0209cf7c63e3faf5008f034002ef61cea93a159a31aa33e18e", "sent_id": "nw_RC00e90a0209cf7c63e3faf5008f034002ef61cea93a159a31aa33e18e-0", "entity_mentions": [{"id": "nw_RC00e90a0209cf7c63e3faf5008f034002ef61cea93a159a31aa33e18e-0-E0", "start": 33, "end": 34, "entity_type": "victim", "mention_type": "UNK", "text": "opponents"}, {"id": "nw_RC00e90a0209cf7c63e3faf5008f034002ef61cea93a159a31aa33e18e-0-E1", "start": 28, "end": 29, "entity_type": "place", "mention_type": "UNK", "text": "Iraq"}], "relation_mentions": [], "event_mentions": [{"event_type": "life:die:n/a", "id": "nw_RC00e90a0209cf7c63e3faf5008f034002ef61cea93a159a31aa33e18e-0-EV0", "trigger": {"start": 40, "end": 41, "text": "exterminated"}, "arguments": [{"entity_id": "nw_RC00e90a0209cf7c63e3faf5008f034002ef61cea93a159a31aa33e18e-0-E0", "text": "opponents", "role": "victim"}, {"entity_id": "nw_RC00e90a0209cf7c63e3faf5008f034002ef61cea93a159a31aa33e18e-0-E1", "text": "Iraq", "role": "place"}]}], "tokens": ["Three", "specific", "points", "illustrate", "why", "Americans", "see", "Trump", "as", "the", "problem", ":", "1", ")", "Trump", "has", "trouble", "working", "with", "people", "beyond", "his", "base", ".", "In", "Saddam", "Hussein", "'s", "Iraq", "that", "might", "work", "when", "opponents", "can", "be", "thrown", "in", "jail", "or", "exterminated", ".", "In", "the", "United", "States", "that", "wo", "n't", "fly", ":", "presidents", "must", "build", "bridges", "within", "and", "beyond", "their", "core", "support", "to", "resolve", "challenges", ".", "Without", "alliances", ",", "a", "president", "ca", "n't", "get", "approval", "to", "get", "things", "done", "."], "pieces": ["Three", "specific", "points", "illustrate", "why", "Americans", "see", "Trump", "as", "the", "problem", ":", "1", ")", "Trump", "has", "trouble", "working", "with", "people", "beyond", "his", "base", ".", "In", "Saddam", "Hussein", "'", "s", "Iraq", "that", "might", "work", "when", "opponents", "can", "be", "thrown", "in", "jail", "or", "ex", "##ter", "##minated", ".", "In", "the", "United", "States", "that", "w", "##o", "n", "'", "t", "fly", ":", "presidents", "must", "build", "bridges", "within", "and", "beyond", "their", "core", "support", "to", "resolve", "challenges", ".", "Without", "alliances", ",", "a", "president", "ca", "n", "'", "t", "get", "approval", "to", "get", "things", "done", "."], "token_lens": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1], "sentence": "Three specific points illustrate why Americans see Trump as the problem : 1 ) Trump has trouble working with people beyond his base . In Saddam Hussein 's Iraq that might work when opponents can be thrown in jail or exterminated . In the United States that wo n't fly : presidents must build bridges within and beyond their core support to resolve challenges . Without alliances , a president ca n't get approval to get things done ."}
```

Note: There are some RAMS data without events (events are extracted from `gold_evt_links`). For example:

```
{"rel_triggers": [], "gold_rel_links": [], "doc_key": "nw_RC013008ad72d04b5e4cca4706cad5cc71c88b0df4615bd597df0e3cf0", "ent_spans": [], "language_id": "eng", "source_url": "http://www.huffingtonpost.com/entry/why-trump-should-peacefully-protest-clintons-victory_us_5809d9b7e4b0b1bd89fdb0bc", "evt_triggers": [[103, 104, [["personnel.elect.winelection", 1.0]]]], "split": "dev", "sentences": [["In", "addition", "to", "working", "alongside", "super", "-", "PACs", ",", "there", "\u2019s", "the", "latest", "saga", "of", "two", "Democratic", "operatives", "losing", "their", "posts", "because", "of", "a", "leaked", "video", "."], ["The", "Chicago", "Tribune", "explains", "the", "impact", "of", "this", "video", "in", "a", "piece", "titled", "Two", "local", "Democratic", "operatives", "lose", "jobs", "after", "video", "sting", "on", "voter", "fraud", ":"], ["Robert", "Creamer", ",", "husband", "of", "Rep.", "Jan", "Schakowsky", ",", "D", "-", "Ill", ".", ",", "and", "Scott", "Foval", "--", "two", "little", "-", "known", "but", "influential", "Democratic", "political", "operatives", "--", "have", "left", "their", "jobs", "after", "video", "investigations", "by", "James", "O'Keefe", "'s", "Project", "Veritas", "Action", "found", "them", "entertaining", "dark", "notions", "about", "how", "to", "win", "elections", "."], ["Foval", "was", "laid", "off", "on", "Monday", "by", "Americans", "United", "for", "Change", ",", "where", "he", "had", "been", "national", "field", "director", "."], ["Creamer", "announced", "Tuesday", "night", "that", "he", "was", "\"", "stepping", "back", "\"", "from", "the", "work", "he", "was", "doing", "for", "the", "unified", "Democratic", "campaign", "for", "Hillary", "Clinton", "."]], "gold_evt_links": []}
```
 
The resultant dataset size after conversion is:
|Data|RAMS|DyGIE++|ONEIE|
|---|---|---|---|
|train|7329|7046|7046
|dev|924|909|909
|test|871|851|851|

### ACE2005 to OneIE format
The `prepreocessing/process_ace.py` script converts raw ACE2005 datasets to the
format used by OneIE. Example:

```
python preprocessing/process_ace.py -i <INPUT_DIR>/LDC2006T06/data -o <OUTPUT_DIR>
  -s resource/splits/ACE05-E -b bert-large-cased -c <BERT_CACHE_DIR> -l english

python preprocessing/process_ace.py -i <PATH_TO_ACE>/data -o input/preprocessed_ace -s <PATH_TO_SPLITS> -l english -b <transformers model, eg. albert-xxlarge-v2> --time_and_val 
```

Arguments:
- -i, --input: Path to the input directory (`data` folder in your LDC2006T06
  package).
- -o, --output: Path to the output directory.
- -b, --bert: Bert model name.
- -c, --bert_cache_dir: Path to the BERT cache directory.
- -s, --split: Path to the split directory. We provide document id lists for all
  datasets used in our paper in `resource/splits`.
- -l, --lang: Language (options: english, chinese).


## Training

- `cd` to the root directory of this package
- Set the environment variable PYTHONPATH to the current directory.
  For example, if you unpack this package to `~/oneie_v0.4.5`, run:
  `export PYTHONPATH=~/oneie_v0.4.5`
- Run this commandline to train a model: `python train.py -c <CONFIG_FILE_PATH>`.
- We provide an example configuration file `config/example.json`. Fill in the 
  following paths in the configuration file:
  - BERT_CACHE_DIR: Pre-trained BERT models, configs, and tokenizers will be 
    downloaded to this directory.
  - TRAIN_FILE_PATH, DEV_FILE_PATH, TEST_FILE_PATH: Path to the training/dev/test
    files.
  - OUTPUT_DIR: The model will be saved to sub folders in this directory.
  - VALID_PATTERN_DIR: Valid patterns created based on the annotation guidelines or training set. Example files are provided in `resource/valid_patterns`.


We may train a new model with `train.py`.  

```
python train.py -c ./config/train_rams.json
```

Training takes many arguments, all of which should be contained in a json file. 
</br>
Here are two key arguments in the json file that should be taken note of.  
1. Location of the train / dev / test data files and the log file. 
```
    "train_file": "<TRAIN_FILE_PATH>",
    "dev_file": "<DEV_FILE_PATH>",
    "test_file": "<TEST_FILE_PATH>",
    "log_path": "<OUTPUT_DIR>",
```
2. Path of the valid patterns.
```
    "valid_pattern_path": "<VALID_PATTERN_DIR>",
```

The files in this path should define specifics of entity / relation / event extraction.  
For instance, `event_role.json` will specify what arguments come with the `Movement:Transport` event.  
```
  "Movement:Transport": [
    "Vehicle",
    "Artifact",
    "Agent",
    "Origin",
    "Destination"
  ],
```

### Training in EAI toolkit
In the root `eai-dsta` directory, run
```
./send_jobs <NAME OF JOB> <TYPE OF JOB>
```
TYPE OF JOB enums: `train`, `test_acepp`

## Evaluation

- `cd` to the root directory of this package
- Set the environment variable PYTHONPATH to the current directory.
  For example, if you unpack this package to `~/oneie_v0.4.5`, run:
  `export PYTHONPATH=~/oneie_v0.4.5`
- Example commandline to use OneIE: `python predict.py -m best.role.mdl -i input -o output -c output_cs --format ltf`
  + Arguments:
    - -m, --model_path: Path to the trained model.
    - -i, --input_dir: Path to the input directory. LTF format sample files can be found in the `input` directory.
    - -o, --output_dir: Path to the output directory (json format). Output files are in the JSON format. Sample files can be found in the `output` directory.
    - -c, --cs_dir: (optional) Path to the output directory (cs format). Sample files can be found in the `output_cs` directory.
    - -l, --log_path: (optional) Path to the log file. A sample file `log.json` can be found in `output`.
    - --gpu: (optional) Use GPU
    - -d, --device: (optional) GPU device index (for multi-GPU machines).
    - -b, --batch_size: (optional) Batch size. For a 16GB GPU, a batch size of 10~15 is a reasonable value.
    - --max_len: (optional) Max sentence length. Sentences longer than this value will be ignored. You may need to decrease `batch_size` if you set `max_len` to a larger number.
    - --beam_size: (optional) Beam set size of the decoder. Increasing this value may improve the results and make the decoding slower.
    - --lang: (optional) Model language.
    - --format: Input file format (txt or ltf).

### Inference 
We can then pass the preprocessed data into the pretrained model with `predict.py`.  
Note that the data needs to be in the specified folder paths.   
```
python predict.py -m models/english.role.v0.3.mdl -i test_train/data/input -o data/output --format json
```  

## Post-processing
To analyse errors from trained models:
```
python main_formatter.py 
   -test R11/test.oneie.json 
   -preds R11/predictions.oneie.json 
   --from_oneie 
   --filter_classes
```

```
python eventspecific.py 
   -preds T4/oneie_formatted_results.jsonl 
   -gold T4/test.json
```



# Output Format

OneIE save results in JSON format. Each line is a JSON object for a sentence 
containing the following fields:
+ doc_id (string): Document ID
+ sent_id (string): Sentence ID
+ tokens (list): A list of tokens
+ token_ids (list): A list of token IDs (doc_id:start_offset-end_offset)
+ graph (object): Information graph predicted by the model
  - entities (list): A list of predicted entities. Each item in the list has exactly
  four values: start_token_index, end_token_index, entity_type, mention_type, score.
  For example, "[3, 5, "GPE", "NAM", 1.0]" means the index of the start token is 3, 
  index of the end token is 4 (5 - 1), entity type is GPE, mention type is NAM,
  and local score is 1.0.
  - triggers (list): A list of predicted triggers. It is similar to `entities`, while
  each item has three values: start_token_index, end_token_index, event_type, score.
  - relations (list): A list of predicted relations. Each item in the list has
  three values: arg1_entity_index, arg2_entity_index, relation_type, score.
  In the following example, `[1, 0, "ORG-AFF", 0.52]` means there is a ORG-AFF relation
  between entity 1 ("leader") and entity 0 ("North Korean") with a local
  score of 0.52.
  The order of arg1 and arg2 can be ignored for "SOC-PER" as this relation is 
  symmetric.
  - roles (list): A list of predicted argument roles. Each item has three values:
  trigger_index, entity_index, role, score.
  In the following example, `[0, 2, "Attacker", 0.8]` means entity 2 (Kim Jong Un) is
  the Attacker argument of event 0 ("detonate": Conflict:Attack), and the local
  score is 0.8.

Output example:
```
{"doc_id": "HC0003PYD", "sent_id": "HC0003PYD-16", "token_ids": ["HC0003PYD:2295-2296", "HC0003PYD:2298-2304", "HC0003PYD:2305-2305", "HC0003PYD:2307-2311", "HC0003PYD:2313-2318", "HC0003PYD:2320-2325", "HC0003PYD:2327-2329", "HC0003PYD:2331-2334", "HC0003PYD:2336-2337", "HC0003PYD:2339-2348", "HC0003PYD:2350-2351", "HC0003PYD:2353-2360", "HC0003PYD:2362-2362", "HC0003PYD:2364-2367", "HC0003PYD:2369-2376", "HC0003PYD:2378-2383", "HC0003PYD:2385-2386", "HC0003PYD:2388-2390", "HC0003PYD:2392-2397", "HC0003PYD:2399-2401", "HC0003PYD:2403-2408", "HC0003PYD:2410-2412", "HC0003PYD:2414-2415", "HC0003PYD:2417-2425", "HC0003PYD:2427-2428", "HC0003PYD:2430-2432", "HC0003PYD:2434-2437", "HC0003PYD:2439-2441", "HC0003PYD:2443-2447", "HC0003PYD:2449-2450", "HC0003PYD:2452-2454", "HC0003PYD:2456-2464", "HC0003PYD:2466-2472", "HC0003PYD:2474-2480", "HC0003PYD:2481-2481", "HC0003PYD:2483-2485", "HC0003PYD:2487-2491", "HC0003PYD:2493-2502", "HC0003PYD:2504-2509", "HC0003PYD:2511-2514", "HC0003PYD:2516-2523", "HC0003PYD:2524-2524"], "tokens": ["On", "Tuesday", ",", "North", "Korean", "leader", "Kim", "Jong", "Un", "threatened", "to", "detonate", "a", "more", "powerful", "H-bomb", "in", "the", "future", "and", "called", "for", "an", "expansion", "of", "the", "size", "and", "power", "of", "his", "country's", "nuclear", "arsenal", ",", "the", "state", "television", "agency", "KCNA", "reported", "."], "graph": {"entities": [[3, 5, "GPE", "NAM", 1.0], [5, 6, "PER", "NOM", 0.2], [6, 9, "PER", "NAM", 0.5060472888322202], [15, 16, "WEA", "NOM", 0.5332313915378754], [30, 31, "PER", "PRO", 1.0], [32, 33, "WEA", "NOM", 1.0], [33, 34, "WEA", "NOM", 0.5212696155645499], [36, 37, "GPE", "NOM", 0.4998288792916457], [38, 39, "ORG", "NOM", 1.0], [39, 40, "ORG", "NAM", 0.5294904130032032]], "triggers": [[11, 12, "Conflict:Attack", 1.0]], "relations": [[1, 0, "ORG-AFF", 1.0]], "roles": [[0, 2, "Attacker", 0.4597024700555278], [0, 3, "Instrument", 1.0]]}}
```
