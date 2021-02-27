from argparse import ArgumentParser
from pathlib import Path
import json

parser = ArgumentParser()
parser.add_argument('-i', '--input', default=None)
args = parser.parse_args()

base_path = Path(Path().absolute().parent)

zero_token_len = False

print(f'Checking {args.input.split("/")[-1]}')

with open(base_path / Path(args.input)) as f:
    data = f.read().splitlines()
    for row in data:
        record = json.loads(row)
        if 0 in record.get('token_lens'):
            print('---')
            print(record)
            print('---')
            zero_token_len = True
            break

if not zero_token_len:
    print('Data is okay.')
else:
    print('Data has zero token len.')

