import os
import json
import requests
from tqdm import tqdm
from vocab import Vocab
from config import Config
from dataset import Dataset, Ontology

'''
Most of the codes are from https://github.com/salesforce/glad
'''

config = Config()

def download(url, to_file):
    r = requests.get(url, stream=True)
    with open(to_file, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

def process_data(lang):
    if not os.path.isdir(os.path.join(config.RAW_DATA_DIR, lang)):
        os.makedirs(os.path.join(config.RAW_DATA_DIR, lang))
    download('https://github.com/nmrksic/neural-belief-tracker/raw/master/data/woz/woz_train_{}.json'.format(lang), os.path.join(config.RAW_DATA_DIR, lang, 'train.json'))
    download('https://github.com/nmrksic/neural-belief-tracker/raw/master/data/woz/woz_validate_{}.json'.format(lang), os.path.join(config.RAW_DATA_DIR, lang, 'dev.json'))
    download('https://github.com/nmrksic/neural-belief-tracker/raw/master/data/woz/woz_test_{}.json'.format(lang), os.path.join(config.RAW_DATA_DIR, lang, 'test.json'))

    print('Processing language: {}'.format(lang))

    splits = ['dev', 'train', 'test']

    dir_to_save_files = os.path.join(config.RAW_DATA_DIR, lang, 'preprocessed')
    # create folder if not exists
    if not os.path.exists(dir_to_save_files):
        os.makedirs(dir_to_save_files)

    #delete any existing file
    for f in os.listdir(dir_to_save_files):
        os.remove(dir_to_save_files + f)

    ontology = Ontology()

    vocab = Vocab()
    vocab.word2index(['pad','sos', 'eos', config.NONE_TOKEN], train=True)

    for s in splits:
        fname = '{}.json'.format(s)
        print('Annotating {}'.format(s))
        dataset = Dataset.annotate_raw(os.path.join(config.RAW_DATA_DIR, lang, fname))
        dataset.numericalize_(vocab)
        
        ontology = ontology + dataset.extract_ontology()
        
        with open(os.path.join(dir_to_save_files, fname), 'wt') as f:
            json.dump(dataset.to_dict(), f)

    ontology.numericalize_(vocab)
    with open(os.path.join(dir_to_save_files, 'ontology.json'), 'wt') as f:
        json.dump(ontology.to_dict(), f)

    with open(os.path.join(dir_to_save_files, 'vocab.json'), 'wt') as f:
        json.dump(vocab.to_dict(), f)



for l in ['en', 'it', 'de']:
    process_data(lang=l)