"""
Generate valid patterns from train data

sample usage:
------------
python scripts/generate_valid_patterns.py \
    -c config/example.json \
    -d valid_patterns_new
"""
import os
import sys
import json
from argparse import ArgumentParser
sys.path.append('.')

from config import Config
from data import IEDataset

def generate_valid_patterns(dataset):
    """
    Generate event_role and role_entity dicts
    
    args:
        dataset: importantly this cannot be numberized
    return:
        event_role, role_entity: dictionaries to be saved in valid patterns
    """
    event_role = {event: [] for event in dataset.event_type_set}
    role_entity = {event: [] for event in dataset.role_type_set}

    try:
        # iterate through each doc
        for doc in dataset:

            # this dictionary maps entity id to its argument rol
            entity_id2type = {entity_mention_dict['id']:entity_mention_dict['entity_type'] 
                            for entity_mention_dict in doc['entity_mentions']}

            # for each role / arg in each event

            for event_mention in doc['event_mentions']:
                event_type = event_mention['event_type']
                for arg_dict in event_mention['arguments']:

                    # save event-role combination
                    role = arg_dict['role']
                    if role not in event_role[event_type]:
                        event_role[event_type].append(role)

                    # # save role-entity combination
                    entity_id = arg_dict['entity_id']
                    entity_type = entity_id2type[entity_id]
                    if entity_type not in role_entity[role]:
                        role_entity[role].append(entity_type)
                
    except:
        print(doc)

    return event_role, role_entity

if __name__ == "__main__":

    # configuration
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', default='config/example.json')
    parser.add_argument('-d', '--dir', default='valid_patterns_new')
    args = parser.parse_args()

    # read config file and dataset
    config = Config.from_json_file(args.config)
    train_set = IEDataset(config.train_file, gpu=False,
                        relation_mask_self=config.relation_mask_self,
                        relation_directional=config.relation_directional,
                        symmetric_relations=config.symmetric_relations,
                        ignore_title=config.ignore_title,
                        max_length=config.sent_max_length)


    # generate valid patterns
    event_role, role_entity = generate_valid_patterns(train_set)

    # make directory
    directory = 'resource/' + args.dir
    if not os.path.isdir(directory): os.makedirs(directory)

    # save files
    with open(directory + '/event_role.json', 'w') as f:
        json.dump(event_role, f, indent=4)
        
    with open(directory + '/role_entity.json', 'w') as f:
        json.dump(role_entity, f, indent=4)
        
    with open(directory + '/relation_entity.json', 'w') as f:
        json.dump({}, f, indent=4)