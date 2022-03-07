import torch
import torch.nn as nn



DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class multitaskNet(nn.Module):
    def __init__(self, encoder, num_heads, sent_len, embed_len, dropout, device, dom=True):
        super(multitaskNet, self).__init__()
        self.device = device
        self.enc = encoder # parameters already defined in training loop
        self.use_dom = dom

        self.sequence_summary = nn.Sequential(
                                            nn.Flatten(), #flatten sequence, heads and embedding dimensions
                                            nn.Linear(sent_len * embed_len, embed_len), # first linear stage compresses sequence dim
                                            nn.Linear(embed_len, num_heads)                         # sencond stage compresses embedding dim
                                            )

        self.fc_1 = nn.Sequential(nn.ReLU(), nn.Linear(num_heads, num_heads), nn.Dropout(dropout))
        self.fc_valence = nn.Sequential(nn.Linear(num_heads, 2), nn.Dropout(dropout), nn.Softmax())
        self.fc_arousal = nn.Sequential(nn.Linear(num_heads, 2), nn.Dropout(dropout), nn.Softmax())
        self.fc_dominance = nn.Sequential(nn.Linear(num_heads, 2), nn.Dropout(dropout), nn.Softmax())

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
        if self.use_dom:
            dominance = self.fc_dominance(out)
            return torch.stack((valence, arousal, dominance))[:, :, 0]
        else:
            return torch.stack((valence, arousal))[:, :, 0]