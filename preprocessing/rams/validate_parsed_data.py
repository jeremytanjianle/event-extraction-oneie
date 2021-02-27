"""
from dygie-rams folder, run:
python scripts/data/rams/validate_parsed_data.py 
"""
import os
from pathlib import Path
from argparse import ArgumentParser
import json

# def validate_doc(doc, idx=''):
#     """
#     Ensure that data follows format specified in:
#     https://github.com/dwadden/dygiepp/blob/master/doc/data.md
#     """
#     # validate that docs have fixed number of entries in "sentences", "ner", "events" fields
#     assert len(doc['sentences']) == len(doc['events']), f": sentences:events = {len(doc['sentences'])}:{ len(doc['events'])}"
#     assert len(doc['sentences']) == 1
            
#     # validate "event" format
#     for list_of_sentences_events in doc['events']:
#         # sent_events = list_of_sentences_events[0]
#         for sent_events in list_of_sentences_events:
#             if len(sent_events[0])>0:
#                 # print(sent_events)
#                 sent_event_trigger = sent_events[0]
#                 sent_event_args = sent_events[1:]
                
#                 assert len(sent_event_trigger) == 2, f"{idx}: trigger expected to have 2 fields, encountered {len(sent_event_trigger)}: {sent_event_trigger}"
#                 for sent_event_arg in sent_event_args:
#                     assert len(sent_event_arg) == 3, f"{idx}: arg expected to have 3 fields, encountered {len(sent_event_arg)}: {sent_event_arg}"

#     return True
from pprint import pprint
# f_list = []
def validate_doc(doc, idx=''):
    # pprint(doc)
    events = doc['events']
    ner = doc['ner']
    relations = doc['relations']
    sentence_start = doc['_sentence_start']
    # sentence_start = doc['sentence_start']
    sentences = doc['sentences']

    if len(set([len(events), len(ner),len(relations), len(sentences), len(sentence_start)])) != 1:
        raise Exception({"events" : len(events), 
                        "ner" : len(ner),
                        "relations" : len(relations), 
                        "sentences" : len(sentences), 
                        "sentence_start" : len(sentence_start)})


    # sentence_single = " ".join([" ".join(sentence) for sentence in sentences])
    sentence_all = []
    for sentence in sentences:
        sentence_all.extend(sentence)

    print(sentence_all)
    for events_in_multiple_sentence in events:
        for events_in_one_sentence in events_in_multiple_sentence:
            arg_span_list = {}
            for event in events_in_one_sentence:
                if len(event) == 2:
                    print(event, sentence_all[event[0]])
                elif len(event) == 3:
                    key = str(event[0]) + "-" + str(event[1])
                    if key in arg_span_list.keys():
                        arg_span_list[key] += 1
                    else:
                        arg_span_list[key] = 1
                    print(event, sentence_all[event[0]:event[1]+1])
            print(arg_span_list)
            # for k, v in arg_span_list.items():
            #     if v > 1:
            #         f_list.append([k, v])
            print("\n")
    print("######################################################")


def validate_docs(docs_file_path):
    with open(docs_file_path, 'r', encoding='utf-8') as r:
        for file_idx, line in enumerate(r):
            doc = json.loads(line)
            validate_doc(doc, idx=file_idx)
    return True

if __name__ == "__main__":

    processed_data_path = Path(os.getcwd()).parent / "oneie/data/rams/processed-data/json"
    for df_json_outfile in ["train.json", "test.json", "dev.json"]:
        validate_docs(processed_data_path / df_json_outfile)
    # print("f list", f_list)
    print("no errors encountered")
