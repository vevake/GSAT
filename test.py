import torch
import numpy as np
import os
import json
from utils import load_dataset, count_parameters
from config import Config
from tqdm import tqdm
from model import Model

config = Config()
dataset, ontology, vocab = load_dataset()

print('Slots: ', ontology.slots)
slot_dict = {s: {'slot_id': idx} for idx, s in enumerate(ontology.slots)}
for s in ontology.slots:
    if s!= 'request':
        slot_dict[s]['values'] = {value: {'value_id':idx, 'num':[vocab.word2index(w) for w in value.split()]} for idx, value in enumerate([config.NONE_TOKEN] + ontology.values[s])}
    else:
        slot_dict[s]['values'] = {value: {'value_id':idx, 'num':[vocab.word2index(w) for w in value.split()]} for idx, value in enumerate(ontology.values[s])}

slot_inv_dict = {sv['slot_id']: {'slot_name': s, 'values': {v['value_id']: val for val, v in sv['values'].items()}} for s, sv in slot_dict.items()}

def get_data(dt, shuffle=False):
    X, Y_label = [], []
    for batch in dt.batch(batch_size=config.BATCH_SIZE, shuffle=shuffle):
        for e in batch:
            a = []
            for act in e.system_acts:
                a.append([vocab.word2index(x.lower()) for x in act if x])
            a = [x for i, s in enumerate(a) for x in s]

            y_true_value  = {}
            for s in slot_dict:
                if s!= 'request':
                    y_true_value[s] = slot_dict[s]['values'][config.NONE_TOKEN]
                else:
                    y_true_value[s] = [0] * len(slot_dict[s]['values'])

            for label in e.turn_label:
                s, v = label
                if s!= 'request':
                    y_true_value[s] = slot_dict[s]['values'][v.lower()]['value_id']
                else:
                    y_true_value[s][slot_dict[s]['values'][v.lower()]['value_id']] = 1

            t = e.num['transcript']

            a = [vocab.word2index('sos')] + a + [vocab.word2index('eos')]            
            X.append(a+t)
            Y_label.append(y_true_value)

    return X, Y_label

   
model = Model(vocab, config.EMBEDDING_DIM, config.HIDDEM_DIM, slot_dict, emb_train=config.TRAIN_EMBEDDING, net=config.RNN, shared_encoder=config.SHARED_ENCODER)
model.cuda()


def eval(data):
    X, Y_label = get_data(data, shuffle=False)
    predictions = [set() for i in range(len(X))]
    model.eval()
    pbar = tqdm(range(0, len(X), config.BATCH_SIZE))
    for b_id in pbar:
        utt = X[b_id:b_id+config.BATCH_SIZE]
        score = model.forward(utt)
        for s in slot_dict:
            if s!= 'request':
                for i, (sc, idx) in enumerate(zip(*score[s].topk(1))):
                    if idx != 0:
                        predictions[b_id+i].add((s, slot_inv_dict[slot_dict[s]['slot_id']]['values'][idx.item()]))
            else:
                score[s] = torch.sigmoid(score[s])
                for i, p in enumerate(score[s]):
                    triggered = [(s, slot_inv_dict[slot_dict[s]['slot_id']]['values'][i], p_v) for i, p_v in enumerate(p) if p_v > config.PROB_THRESHOLD]
                    predictions[b_id+i] |= set([(s, v) for s, v, p_v in triggered])
    model.train()
    return data.evaluate_preds(predictions), predictions

print('loading model...')
model.load_state_dict(torch.load(os.path.join(config.SAVE_DIR,'best.t7')))
print('loaded.')
dev_result, dev_predictions = eval(dataset['dev'])
print('Dev: {}'.format(dev_result))
test_result, test_predictions = eval(dataset['test'])
print('Test: {}'.format(test_result))
dataset['test'].record_preds(test_predictions, os.path.join(config.SAVE_DIR, 'prediction_test.json'))