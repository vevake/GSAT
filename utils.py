import json
import os
from pprint import pformat
from vocab import Vocab
from config import Config
from dataset import Dataset, Ontology

'''
Most of the codes are from https://github.com/salesforce/glad
'''

config = Config()

def load_dataset(emb=False, splits=('train', 'dev', 'test')):
    with open(os.path.join(config.DATA_DIR, 'ontology.json')) as f:
        ontology = Ontology.from_dict(json.load(f))
    
    with open(os.path.join(config.DATA_DIR, 'vocab.json')) as f:
        vocab = Vocab.from_dict(json.load(f))
    
    if emb:
        with open(os.path.join(config.DATA_DIR, 'emb.json')) as f:
            E = json.load(f)
        
    dataset = {}
    for split in splits:
        with open(os.path.join(config.DATA_DIR, '{}.json'.format(split))) as f:
            dataset[split] = Dataset.from_dict(json.load(f))

    print('dataset sizes: {}'.format(pformat({k: len(v) for k, v in dataset.items()})))

    if emb:
        return dataset, ontology, vocab, E
    else:
        return dataset, ontology, vocab


def count_parameters(model):
    c = 0
    for name, p in model.named_parameters():
        c += p.numel()
    return sum(p.numel() for p in model.parameters() if p.requires_grad)