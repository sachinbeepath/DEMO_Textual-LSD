import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchtext.legacy.data import Field
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import argparse
import spacy
import utils

import dataloader as dl
import transformer as tr
import multitaskNet as mtn

'''
This is the main training script.
Contains the primary training loop for the Textual-LSD research network.
'''

##### Key Variables #####
EPOCHS = 10000
BATCH_SIZE = 32
LR = 3e-4
USE_DOM = True
FILENAME = 'Data_8500_songs.xlsx'
ATTENTION_HEADS = 8
EMBEDDING_SIZE = 512
NUM_ENCODER_LAYERS = 3
FORWARD_XP = 4
DROPOUT = 0.25
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TRAIN_VAL_SPLIT = 0.8   

##### Load Data #####
file = pd.read_excel(FILENAME)
dataframe = pd.DataFrame(file)

dataset = dl.LSD_DataLoader(dataframe, 'lyrics', ['valence_tags', 'arousal_tags', 'dominance_tags'])
idx = np.random.shuffle(np.arange(0, len(dataset), 1))
dataset_tr = Subset(dataset, idx[:int(TRAIN_VAL_SPLIT * len(idx))])
dataset_val = Subset(dataset, idx[int(TRAIN_VAL_SPLIT * len(idx)):])
dataloader_tr = DataLoader(dataset_tr, batch_size=BATCH_SIZE, shuffle=True)
dataloader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=True)

# Vocab work
VOCAB = spacy.load('en')
english = Field(init_token='<SOS>', eos_token='<EOS>')
english.build_vocab(dataframe['lyrics'][idx[:int(TRAIN_VAL_SPLIT * len(idx))]], max_size=30000, min_freq=2)
PAD_IDX = english.vocab.stoi['<PAD>']
VOCAB_LEN = len(english.vocab)

##### Prepare Model, Optimizer and Criterion #####
encoder = None
multitask = mtn.multitaskNet(encoder, ATTENTION_HEADS, dataset.length, EMBEDDING_SIZE, DROPOUT, USE_DOM).to(device=DEVICE)

adam = optim.Adam(multitask.parameters(), lr=LR) # Fine tune this hypPs...

valence_L = nn.MSELoss()
arousal_L = nn.MSELoss()
dominance_L = nn.MSELoss()

losses = []
# Training Loop
for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1} / {EPOCHS}')
    multitask.eval()
    
    for batch_idx, batch in enumerate(dataloader_tr):
        inp_data = utils.text_to_index(batch['lyrics'])
        val = batch['valence_tags']
        aro = batch['arousal_tags']
        dom = batch['dominance_tags']

        output = multitask(torch.tensor(inp_data))

        adam.zero_grad()

        valence_loss = valence_L(output[0], val)
        arousal_loss = arousal_L(output[1], aro)
        dominance_loss = dominance_L(output[2], dom)
        loss = valence_loss + arousal_loss + dominance_loss
        loss.backward()
        torch.nn.utils.clip_grad(multitask.parameters(), max_norm=1)
        adam.step()

        losses.append(loss)

torch.save(multitask.state_dict(), 'Model.pt')

print('Done')






