import os
import json
from tqdm import tqdm
from argparse import ArgumentParser

# configuration
parser = ArgumentParser()
parser.add_argument("-i", default=None, type=str)
parser.add_argument("-o", default=None, type=str)
args = parser.parse_args()


def split_article_sentences(article, article_idx):
    sentences = article.get('sentences')
    splitted_sentence = {
        'doc_id': article_idx
    }
    new_article = dict(article)
    new_article.pop('sentences', None)
    new_sentences = []
    for idx, sentence in enumerate(sentences):
        splitted_sentence['sent_id'] = f'{splitted_sentence["doc_id"]}-{idx}'
        splitted_sentence['tokens'] = sentence
        splitted_sentence.update(new_article)
        new_sentences.append(dict(splitted_sentence))
    return new_sentences


if __name__ == "__main__":
    # load cc json
    with open(args.i, 'r') as f, open(args.o, 'w') as output_f:
        articles = [json.loads(a) for a in f.readlines()]

        # iterate through each article
        for idx, article in tqdm(enumerate(articles)):
            splitted_sentences = split_article_sentences(article, idx)

            for sent in splitted_sentences:
                # print(sent)
                # save the sentences back into a json
                json.dump(sent, output_f)
                output_f.write('\n')
