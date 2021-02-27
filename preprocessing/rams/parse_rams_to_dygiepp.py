import re
import os
import jsonlines
import pandas as pd
from tqdm import tqdm
import itertools

from pathlib import Path

def read_jsonlines(path):
    data = []
    with jsonlines.open(path) as reader:
        for obj in reader:
            data.append(obj)
    return data

raw_data_path = Path(os.getcwd()).parent / "oneie/data/rams/raw/RAMS_1.0/data"
processed_data_path = Path(os.getcwd()).parent / "oneie/data/rams/processed-data/json"

train_df = pd.DataFrame(read_jsonlines(raw_data_path / "train.jsonlines"))
val_df = pd.DataFrame(read_jsonlines(raw_data_path / "dev.jsonlines"))
test_df = pd.DataFrame(read_jsonlines(raw_data_path / "test.jsonlines"))

def process_row(row):
    # For our purposes, we require "sentences" and "events" fields

    # concatenate the 5 sentences into 1 big sentence as specified in 
    # https://github.com/dwadden/dygiepp/issues/38

    concat_sentence=[]
    for i, sentence in enumerate(row["sentences"]):
        concat_sentence+=sentence
    row['sentences'] = [concat_sentence]     

    ## Create list for events and add one sublist for each sentence. Note that 
    # sentence with no ner or event will have an emtpy sublist. This is to create the 
    # structure required by Dygiepp
    events = []
    
    # Create events from evt_triggers and gold_evt_links. Note that each sample have 
    # one and only one evt_triggers, and each trigger can only consist of one token
    # sample - "evt_triggers": [[40, 40, [["life.die.n/a", 1.0]]]]
    evt_trigger_start = row['evt_triggers'][0][0]
    evt_trigger_end = row['evt_triggers'][0][1]
    evt_trigger_type = row['evt_triggers'][0][-1][0][0]
    events.append([evt_trigger_start, evt_trigger_type])
    ## Debug
    #print(evt_trigger_start, evt_trigger_end, evt_trigger_type)
    
    # Check to make sure trigger is only one token
    if evt_trigger_start == evt_trigger_end:
        for evt_links in row['gold_evt_links']:
            # sample of "gold_evt_links"
            # [
            #     [[69, 69], [85, 88], "evt090arg01killer"], 
            #     [[69, 69], [42, 43], "evt090arg02victim"], 
            #     [[69, 69], [26, 26], "evt090arg04place"]
            # ]
            # evt_trigger = evt_links[0]
            ent_span = evt_links[1]
            ent = evt_links[2][11:]
            ## Debug
            #print(ent_span, ent)

            ent_span.append(ent)
            #print(ent_span)
            if len(ent_span)>3:
                ent_span = ent_span[:3]
            events.append(ent_span)
            #print(events)
        row['events'] = [[events]]
        # print(row['events'])
    else:
        row['events'] = [[]]

    # Initialise list for NER
    new_entity_list = []
    for relation in row['gold_evt_links']:
        # collation of ner
        ner_input = relation[1][:-1]+[relation[1][-1]]
        if len(ner_input)>3:
            ner_input=ner_input[:3]
        new_entity_list.append(ner_input)

    row['ner'] = [new_entity_list]

    # RAMS does not have 'relations' tags
    row['relations'] = [[]]

    # Append the start position of our sentence into a sentence_start list (1st sentence = 0) 
    row['_sentence_start'] = [0] * 1
    row['dataset'] = "rams"
    return row


def process_df(df):
    print(f'Original dataframe shape: {df.shape}')
    df_new = df[["ent_spans", "evt_triggers", "sentences", "gold_evt_links", "doc_key"]].copy()
    df_new = df_new.apply(lambda row: process_row(row), axis=1)
    df_new = df_new[["doc_key", "sentences", "events", "ner", "relations", "_sentence_start", "dataset"]]
    
    # Remove entries where the events/entities return an empty label
    no_event_count = 0
    for idx, row in tqdm(df_new.iterrows()):
        if row['events']==[[]]:
            no_event_count += 1
            df_new.drop(idx, axis=0, inplace=True)

    print(f'Removing {no_event_count} rows without events.')

    # df_new = df_new[df_new["events"].notnull()]
    print(f'Processed dataframe.shape: {df_new.shape}')
    return df_new
#for df, df_json_outfile in [(train_df[:10], "train.json"), (test_df[:10], "test.json"), (val_df[:10], "dev.json")]:
for df, df_json_outfile in [(train_df, "train.json"), (test_df, "test.json"), (val_df, "dev.json")]:
    print(f'=== Reading "{df_json_outfile}"')
    df_new = process_df(df)
    df_new.to_json(str(Path(processed_data_path / df_json_outfile)), orient='records', lines=True)
