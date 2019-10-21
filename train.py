import numpy as np
import os
import json
import random
from config import Config
from model import Model
from tqdm import tqdm
from utils import load_dataset, count_parameters
import torch
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F

config = Config()
print(config)
#set random seed for experiment
print('Seed: {}'.format(config.SEED))
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)
torch.cuda.manual_seed(config.SEED)
torch.cuda.manual_seed_all(config.SEED)
random.seed(config.SEED)

#create SAVE_DIR if not exists
if not os.path.isdir(config.SAVE_DIR):
    os.makedirs(config.SAVE_DIR)

#load dataset
if config.USE_PRETRAINED_EMB:
    dataset, ontology, vocab, Eword = load_dataset(emb=True)
else:
    dataset, ontology, vocab = load_dataset()

#Dictionary for slot_lables
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
                    y_true_value[s] = slot_dict[s]['values'][config.NONE_TOKEN]['value_id']
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

if config.USE_PRETRAINED_EMB:
    model.load_emb(Eword.copy())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)


print('Number of parameters:{}'.format(count_parameters(model)))
    
best_dev = {}
best_epoch = 0

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


epoch = 0
while epoch < config.MAX_EPOCH:
    print('Starting epoch {}'.format(epoch+1))
    model.train()
    X, Y_label = get_data(dataset['train'], shuffle=True)
    total_loss = 0.
    cnt = 1
    pbar = tqdm(range(0, len(X), config.BATCH_SIZE))
    for b_id in pbar:
        model.zero_grad()
        utt = X[b_id:b_id+config.BATCH_SIZE]
        logits = model.forward(utt)
        score = {s: Variable(torch.Tensor(len(utt), len(slot_dict[s]['values'])).zero_().cuda()) for s in slot_dict}
        for s in slot_dict:
            score[s] += logits[s]
        
        labels = {}
        for s in slot_dict:
            if s!='request':
                labels[s] = [0]*len(utt)
            else:               
                labels[s] = [[0] * len(slot_dict[s]['values'])] * len(utt)

        for s in slot_dict:
            for i, y in enumerate(Y_label[b_id:b_id+config.BATCH_SIZE]):
                labels[s][i] = y[s]
        
        for s, m in labels.items():
            if s!='request':
                labels[s]= torch.LongTensor(m).cuda()
            else:
                labels[s]= torch.Tensor(m).cuda()
        
        loss = {s: 0 for s in slot_dict}
        for s in ontology.slots:
            if s!='request':
                loss[s] += F.cross_entropy(score[s], labels[s]) 
            else:
                loss[s] += F.binary_cross_entropy_with_logits(score[s], labels[s])
                
        loss = sum(loss.values())
        total_loss += loss.item()
        pbar.set_description(desc='Epoch:{}, L:{:.4f}'.format(epoch+1, total_loss/cnt))
        cnt += 1
        loss.backward()
        optimizer.step()
    dev_result, dev_predictions = eval(dataset['dev'])
    print('Dev: {}'.format(dev_result))
    if best_dev.get(config.METRIC_TO_EVALUATE, 0) < dev_result[config.METRIC_TO_EVALUATE]:
        best_dev = dev_result
        best_epoch = epoch + 1
        torch.save(model.state_dict(), os.path.join(config.SAVE_DIR, 'best.t7'))
        if int(epoch*1.2) > config.MAX_EPOCH:
            max_epoch = int(epoch*1.2)

    print('Best dev: {} in epoch {}'.format(best_dev.get(config.METRIC_TO_EVALUATE, 0), best_epoch))
    epoch += 1
print('\Best model result.')
print('Dev: {}'.format(best_dev))
