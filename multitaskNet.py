import torch
import torch.nn as nn



DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class multitaskNet(nn.Module):
    def __init__(self, encoder, num_heads, sent_len, embed_len, dropout, dom=True):
        super(multitaskNet, self).__init__()
        self.trans = encoder
        self.use_dom = dom

        self.sequence_summary = nn.Sequential(
                                            nn.Flatten(), #flatten sequence, heads and embedding dimensions
                                            nn.Linear(num_heads * sent_len * embed_len, num_heads * embed_len), # first linear stage compresses sequence dim
                                            nn.Linear(num_heads * embed_len, num_heads)                         # sencond stage compresses embedding dim
                                            )

        self.fc_1 = nn.Sequential(nn.Relu(), nn.Linear(num_heads, num_heads), nn.Dropout(dropout))
        self.fc_valence = nn.Sequential(nn.Linear(num_heads, 2), nn.Dropout(dropout), nn.Softmax())
        self.fc_arousal = nn.Sequential(nn.Linear(num_heads, 2), nn.Dropout(dropout), nn.Softmax())
        self.fc_dominance = nn.Sequential(nn.Linear(num_heads, 2), nn.Dropout(dropout), nn.Softmax())

    def forward(self, input_ids, token_ids=None, labels=None):
        '''
        transformer needs to output dimension: BxLxHxE (sum out the embedding dim)
        B = batch size, L = sentence length, H = number of attention heads, E = embedding size
        '''

        out = self.trans(input_ids, token_ids, labels)  #BxLxHxE
        out = self.sequence_summary(out)                #BxH
        out = self.fc_1(out)                            #BxH
        valence = self.fc_valence(out)                  #Bx2 for the rest
        arousal = self.fc_arousal(out)
        if self.use_dom:
            dominance = self.fc_dominance(out)
            return torch.cat((valence, arousal, dominance), axis=1)
        else:
            return torch.cat((valence, arousal), axis=1)