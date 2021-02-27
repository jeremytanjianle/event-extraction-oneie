from pathlib import Path
import json

base_path = Path(__file__).parents[1]

vp_rams_folder = base_path / 'resource/valid_patterns_rams'
vp_ace_folder = base_path / 'resource/valid_patterns_ace'
vp_acepp_folder = base_path / 'resource/valid_patterns_acepp'

##########################################################
# process event_role.json
##########################################################
with open(vp_rams_folder / 'event_role.json') as f:
    rams_event_role = json.load(f)
with open(vp_ace_folder / 'event_role.json') as f:
    ace_event_role = json.load(f)


def preprocess_event_role(d, t):
    keys = list(d.keys())
    for k in keys:
        if t == 'rams':
            # remove one layer of events for rams data
            new_k = ':'.join(k.split(':')[:-1])
        else:
            # lowercase and strip '-' of events for ace data
            new_k = k.replace('-', '').lower()
        if d.get(new_k) is not None:
            l = d.get(new_k)
            l.extend(d.pop(k))
            new_value = list(set(l))
            d[new_k] = new_value
        else:
            d[new_k] = d.pop(k)


preprocess_event_role(rams_event_role, 'rams')
preprocess_event_role(ace_event_role, 'ace')

# merge rams and ace event_role
merged_list = list(rams_event_role.keys())
merged_list.extend(list(ace_event_role.keys()))
merged_keys = set(merged_list)
acepp_event_role = {}
for k in merged_keys:
    acepp_l = []
    if rams_event_role.get(k) is not None:
        acepp_l.extend(rams_event_role.get(k))

    if ace_event_role.get(k) is not None:
        acepp_l.extend(ace_event_role.get(k))

    acepp_l = [x.lower() for x in acepp_l]
    acepp_event_role[k] = list(set(acepp_l))

with open(vp_acepp_folder/'event_role.json', 'w') as f:
    json.dump(acepp_event_role, f)


##########################################################
# process event_role.json
##########################################################
with open(vp_rams_folder / 'role_entity.json') as f:
    rams_role_entity = json.load(f)
with open(vp_ace_folder / 'role_entity.json') as f:
    ace_role_entity = json.load(f)


def preprocess_role_entity(d):
    keys = list(d.keys())
    for k in keys:
        new_k = k.lower()
        d[new_k] = d.pop(k)


preprocess_role_entity(ace_role_entity)
# merge rams and ace event_role
merged_list = list(rams_role_entity.keys())
merged_list.extend(list(ace_role_entity.keys()))
merged_keys = set(merged_list)
acepp_role_entity = {}
for k in merged_keys:
    acepp_l = []
    if rams_role_entity.get(k) is not None:
        acepp_l.extend(rams_role_entity.get(k))

    if ace_role_entity.get(k) is not None:
        acepp_l.extend(ace_role_entity.get(k))

    acepp_role_entity[k] = list(set(acepp_l))

with open(vp_acepp_folder/'role_entity.json', 'w') as f:
    json.dump(acepp_role_entity, f)

