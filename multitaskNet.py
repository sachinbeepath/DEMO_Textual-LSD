from audioop import bias
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer as trans

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class multitaskNet(nn.Module):
    def __init__(self, hidden_size, sent_len, embed_len, dropout, device, vocab_size, num_layers, att_heads, mult, dom=True):
        super(multitaskNet, self).__init__()
        self.device = device
        self.enc = trans.Encoder(vocab_size, embed_len, num_layers, att_heads, mult, dropout, sent_len, device)
        self.enc.double()
        self.use_dom = dom

        self.sequence_summary = nn.Sequential(
                                            nn.Flatten(), #flatten sequence, heads and embedding dimensions
                                            nn.Linear(sent_len * embed_len, embed_len), # first linear stage compresses sequence dim
                                            nn.ReLU(),
                                            nn.Linear(embed_len, 2 * hidden_size)                         # sencond stage compresses embedding dim
                                            )
        
        self.fc_1 = nn.Sequential(nn.ReLU(), nn.Dropout(dropout), nn.Linear(2 * hidden_size, hidden_size))
        self.fc_valence = nn.Linear(hidden_size, 2, bias=False)
        self.fc_arousal = nn.Linear(hidden_size, 2, bias=False)
        self.fc_dominance = nn.Linear(hidden_size, 2, bias=False)
        self.fc_quad = nn.Sequential(nn.ReLU(), nn.Linear(hidden_size, 4))

    def forward(self, x):
        '''
        transformer needs to output dimension: BxLxHxE (sum out the embedding dim)
        B = batch size, L = sentence length, H = number of attention heads, E = embedding size
        '''

        out = self.enc(x)  #BxLxHxE
        out = self.sequence_summary(out)                #BxH
        out = self.fc_1(out)                            #BxH
        valence = self.fc_valence(out)                  #Bx2 for the rest
        arousal = self.fc_arousal(out)
        quad = self.fc_quad(out)
        if self.use_dom:
            dominance = self.fc_dominance(out)
            return torch.stack((valence, arousal, dominance)), quad
        else:
            return torch.stack((valence, arousal)), quad