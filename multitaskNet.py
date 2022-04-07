import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer as trans
from torch.autograd import Variable

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class multitaskNet(nn.Module):
    def __init__(self, hidden_size, sent_len, embed_len, dropout, device, vocab_size, num_layers, att_heads, mult,
                 pad_idx, dom=True, w2v=None):
        super(multitaskNet, self).__init__()
        self.device = device
        self.pos_emb = nn.Embedding(sent_len, embed_len)
        if w2v is not None:
            self.word_emb = nn.Embedding(vocab_size, embed_len, pad_idx)
            self.word_emb.load_state_dict({'weight' : w2v})
        else:
            self.word_emb = nn.Embedding(vocab_size, embed_len, pad_idx)
        self.enc_manual = trans.Encoder(vocab_size, embed_len, num_layers, att_heads, mult, dropout, sent_len, device, w2v)
        self.enc_manual.double()
        enc_layer = nn.TransformerEncoderLayer(embed_len, att_heads, mult, dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers)
        # self.pretrained_trans = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased", num_labels=embed_len)
        self.use_dom = dom
        self.dropout = nn.Dropout(dropout)
        self.flt = nn.Flatten()
        self.sequence_summary = nn.Sequential(
                                            nn.Linear(sent_len * embed_len, embed_len), # first linear stage compresses sequence dim
                                            nn.ReLU(),
                                            nn.Linear(embed_len, 2 * hidden_size)                         # sencond stage compresses embedding dim
                                            )
        self.sequence_summary_2 = nn.Linear(embed_len, 2 * hidden_size)
        
        self.fc_1 = nn.Sequential(nn.ReLU(), nn.Dropout(dropout), nn.Linear(2 * hidden_size, hidden_size))
        self.fc_valence = nn.Linear(hidden_size, 2, bias=False)
        self.fc_arousal = nn.Linear(hidden_size, 2, bias=False)
        self.fc_dominance = nn.Linear(hidden_size, 2, bias=False)
        self.fc_quad = nn.Sequential(nn.ReLU(), nn.Linear(hidden_size, 4))

    def forward(self, x, version):
        '''
        transformer needs to output dimension: BxLxHxE (sum out the embedding dim)
        B = batch size, L = sentence length, H = number of attention heads, E = embedding size
        '''
        if version == 0:
            #Create embeddings first
            N, seq_length = x.shape
            positions = torch.arange(0,seq_length).expand(N,seq_length).to(self.device)
            x = self.dropout(self.word_emb(x) + self.pos_emb(positions))
            #Now run transformer encoders
            out = self.flt(self.enc(x))  #BxLxHxE
            out = self.sequence_summary(out) 
        elif version == 1:
            out = self.flt(self.enc_manual(x))
            out = self.sequence_summary(out) 

        elif version == 2:
            with torch.no_grad():
                out = self.pretrained_trans(x).logits
            out = self.sequence_summary_2(out) 
                       #BxH
        out = self.fc_1(out)                            #BxH
        valence = self.fc_valence(out)                  #Bx2 for the rest
        arousal = self.fc_arousal(out)
        quad = self.fc_quad(out)
        if self.use_dom:
            dominance = self.fc_dominance(out)
            return torch.stack((valence, arousal, dominance)), quad
        else:
            return torch.stack((valence, arousal)), quad

class BiLSTMSentiment(nn.Module):

    def __init__(self, hidden_size, sent_len, embed_len, dropout, device, vocab_size, pad_idx, dom=True, w2v=None):
        super(BiLSTMSentiment, self).__init__()
        self.device = device
        self.batch_size = sent_len
        self.hidden_size = hidden_size
        if w2v is not None:
            self.embeddings = nn.Embedding(vocab_size, embed_len, pad_idx)
            self.embeddings.load_state_dict({'weight': w2v})
        else:
            self.embeddings = nn.Embedding(vocab_size, embed_len, pad_idx)
        self.lstm = nn.LSTM(input_size=embed_len, hidden_size=hidden_size, bidirectional=True, dropout=dropout, batch_first=True,
                            num_layers=2, device=device)
        self.hidden = self.init_hidden(self.batch_size)
        self.use_dom = dom
        self.dropout = nn.Dropout(dropout)
        self.flt = nn.Flatten()
        self.fc_1 = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden_size, hidden_size), nn.Tanh())
        self.fc_valence = nn.Linear(hidden_size, 2, bias=False)
        self.fc_arousal = nn.Linear(hidden_size, 2, bias=False)
        self.fc_dominance = nn.Linear(hidden_size, 2, bias=False)
        self.fc_quad = nn.Sequential(nn.Linear(hidden_size, 4))

    def init_hidden(self,batch_size):
        # first is the hidden h
        # second is the cell c
        hid = (torch.zeros(4, batch_size, self.hidden_size).double().to(self.device),
               torch.zeros(4, batch_size, self.hidden_size).double().to(self.device),)
        return hid

    def forward(self, x, version):
        '''
        transformer needs to output dimension: BxLxHxE (sum out the embedding dim)
        B = batch size, L = sentence length, H = number of attention heads, E = embedding size
        '''
        self.hidden = self.init_hidden(x.shape[0])
        x = self.embeddings(x)#.view(len(x),self.batch_size,-1)
        out, _ = self.lstm(x.double(),self.hidden)
        out = self.fc_1(out[:,-1,:])                            #BxH
        valence = self.fc_valence(out)                  #Bx2 for the rest
        arousal = self.fc_arousal(out)
        quad = self.fc_quad(out)
        if self.use_dom:
            dominance = self.fc_dominance(out)
            return torch.stack((valence, arousal, dominance)), quad
        else:
            return torch.stack((valence, arousal)), quad
