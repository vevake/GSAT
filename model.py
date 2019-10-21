import numpy as np
import math
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F


class EmbeddingLayer(nn.Embedding):
    def __init__(self, *args, dropout=0.2, emb_train=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.dropout = dropout
        self.emb_train = emb_train

    def forward(self, *args, **kwargs):
        out = super().forward(*args, **kwargs)
        if not self.emb_train:
            out.detach_()
            
        return F.dropout(out, self.dropout, self.training)

class Attn(nn.Module):
    def __init__(self, inp_size, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(inp_size + self.hidden_size, hidden_size)
        self.v = nn.Parameter(torch.zeros(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, encoder_outputs, hidden, lengths=None, normalize=True):
        attn_energies = self.score(hidden, encoder_outputs)
        
        attn_energies_ = attn_energies.clone()
        if lengths is not None:
            max_len = max(lengths)
            for i, lp in enumerate(lengths):
                if lp < max_len:
                    attn_energies_[i, :, lp:] = 0.
                    attn_energies[i, :, lp:] = float('-inf')
        
        normalized_energy = nn.Softmax(dim=2)(attn_energies)
        context = torch.bmm(normalized_energy, encoder_outputs)
        return context

    def score(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(1)
        H = hidden.unsqueeze(1).expand_as(encoder_outputs)
        energy = nn.Tanh()(self.attn(torch.cat([H, encoder_outputs], 2)))
        energy = energy.transpose(2, 1)
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)
        energy = torch.bmm(v, energy)
        return energy
    
def init_gru(gru):
    gru.reset_parameters()
    for _, hh, _, _ in gru.all_weights:
        for i in range(0, hh.size(0), gru.hidden_size):
            torch.nn.init.orthogonal_(hh[i:i+gru.hidden_size],gain=1)
    
class UtteranceEncoder(nn.Module):
    def __init__(self, embed_size, hidden_size, dropout=0.2, net='gru'):
        super(UtteranceEncoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p=dropout)
        self.attn = Attn(2*hidden_size, 2*hidden_size)
        
        self.net_type = net.lower()
        if self.net_type == 'gru':
            self.net = nn.GRU(input_size=self.embed_size, hidden_size=hidden_size, bidirectional=True, batch_first=True)
        elif self.net_type == 'lstm':
            self.net = nn.LSTM(input_size=self.embed_size, hidden_size=hidden_size, bidirectional=True, batch_first=True)
        else:
            print('Invalid network type.')
        init_gru(self.net)
        
    def forward(self, utterance, utt_len):
        input_lens = np.array(utt_len)
        sort_idx = np.argsort(-input_lens)
        unsort_idx = torch.LongTensor(np.argsort(sort_idx)).cuda()
        input_lens = input_lens[sort_idx]
        sort_idx = torch.LongTensor(sort_idx).cuda()
        embedded = utterance[sort_idx]
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lens, batch_first=True)
        
        if self.net_type == 'gru':
            out, hidden = self.net(packed)
        
        elif self.net_type == 'lstm':
            out, (hidden, ctx) = self.net(packed)            
        
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        out = out[unsort_idx]
        hidden = hidden.transpose(0,1)[unsort_idx]
        
        return out, hidden.contiguous().view(utterance.size(0), -1)
    
class SlotDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.2, inf=True):
        super(SlotDecoder, self).__init__()
        self.attn = Attn(input_dim, input_dim)
        self.informable = inf
        if self.informable:
            self.output_dim = output_dim-1
            self.none_score = nn.Parameter(torch.Tensor([0.2]))
        else:
            self.output_dim = output_dim
            self.linear_output = nn.Linear(input_dim, self.output_dim)
        self.linear_input = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.tanh = nn.ReLU()
        self.linear_values = nn.Linear(input_dim, input_dim)

    def forward(self, inp_t, inp_n, inp_len, slot_values):
        inp_n = self.dropout(inp_n)
        inp_t = self.dropout(inp_t)
        inp_n = self.tanh(self.linear_input(inp_n))
        inp_n = self.dropout(inp_n)
        inp_n = self.attn(inp_t, inp_n, lengths=inp_len)
        if self.informable:
            score = torch.mm(inp_n.squeeze(1), self.dropout(self.linear_values(slot_values[1:])).transpose(1,0))
            score = torch.cat([self.none_score.expand(inp_t.size(0), 1), score], dim=-1)
        else:
            score = torch.mm(inp_n.squeeze(1), self.dropout(self.linear_values(slot_values)).transpose(1,0))
            # score = self.linear_output(inp_n.squeeze(1))
                
        return score

def pad_utterance(seqs, emb, device, pad=0):
    lens = [len(s) for s in seqs]
    max_len = max(lens)
    padded = torch.LongTensor([s + (max_len-l) * [pad] for s, l in zip(seqs, lens)])
    return emb(padded.to(device)), lens

class Model(nn.Module):
    def __init__(self, vocab, emb_dim, enc_hidden_dim, slot_dict, dropout=0.2, emb_train=True, net='gru', shared_encoder=False):
        super(Model, self).__init__()
        self.vocab = vocab
        self.emb_fixed = EmbeddingLayer(len(vocab), emb_dim, dropout=dropout, emb_train=emb_train)
        self.slot_dict = slot_dict
        self.shared_encoder = shared_encoder
        
        if self.shared_encoder:
            self.utterance_encoder = UtteranceEncoder(emb_dim, enc_hidden_dim, dropout, net=net)
        else:            
            utterance_encoder = {}
            for s in self.slot_dict:
                utterance_encoder[s] = UtteranceEncoder(emb_dim, enc_hidden_dim, dropout, net=net)
            self.utterance_encoder = utterance_encoder
            self.utterance_encoder_list = nn.ModuleList(self.utterance_encoder.values())        
                    
        slot_decoder = {}
        for s in self.slot_dict:
            if s!='request':
                slot_decoder[s] = SlotDecoder(2*enc_hidden_dim, len(self.slot_dict[s]['values']), dropout=dropout, inf=True)
            else:
                slot_decoder[s] = SlotDecoder(2*enc_hidden_dim, len(self.slot_dict[s]['values']), dropout=dropout, inf=False)
        self.slot_decoder = slot_decoder
        self.slot_decoder_list = nn.ModuleList(self.slot_decoder.values())
        
    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_emb(self, Eword):
        new = self.emb_fixed.weight.data.new
        self.emb_fixed.weight.data.copy_(new(Eword))
        
    def forward(self, utt):
        pad_id = self.vocab.word2index('pad')
        
        utt_emb, utt_len = pad_utterance(utt, self.emb_fixed, self.device, pad=pad_id)
                            
        if self.shared_encoder:
            enc_out = self.utterance_encoder(utt_emb, utt_len)
        else:
            enc_out = {s: self.utterance_encoder[s](utt_emb, utt_len) for i, s in enumerate(self.slot_dict)}
        slot_values = {s: pad_utterance([v['num'] for _, v in self.slot_dict[s]['values'].items()], self.emb_fixed, self.device, pad=pad_id)[0] for s in self.slot_dict}
        
        slot_values = {s: torch.sum(slot_values[s], dim=1) for s in self.slot_dict}
        
        if self.shared_encoder:
            scores = {s: self.slot_decoder[s](enc_out[0], enc_out[1], utt_len, slot_values[s]) for s in self.slot_dict}
        else:
            scores = {s: self.slot_decoder[s](enc_out[s][0], enc_out[s][1], utt_len, slot_values[s]) for i, s in enumerate(self.slot_dict)}
                        
        return scores