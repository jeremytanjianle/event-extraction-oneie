from argparse import ArgumentParser
from pathlib import Path
import json

parser = ArgumentParser()
parser.add_argument('-i', '--input', default=None)
args = parser.parse_args()

base_path = Path(Path().absolute().parent)

print(f'Getting conflict events from {args.input.split("/")[-1]}')

print(base_path / Path(args.input))

output_filename = args.input.split("/")[-1].split('.')
output_filename[0] = f'{output_filename[0]}-conflict'
output_filename = '.'.join(output_filename)
output_path = Path(f'{"/".join(args.input.split("/")[:-1])}/{output_filename}')

with open(base_path / Path(args.input)) as f, open(base_path / output_path, 'w+') as output_f:
    data = [json.loads(x) for x in f.read().splitlines()]
    data_events = [x for x in data if x.get('event_mentions') != []]
    conflict_data = [x for x in data_events if 'conflict' in x.get('event_mentions')[0].get('event_type')]
    print(f'Saving {len(conflict_data)} conflict event records.')
    for d in conflict_data:
        json.dump(d, output_f)
        output_f.write("\n")
    print('Done.')
