import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import argparse
import utils
import os

clear = lambda: os.system('cls')

def printc(text):
    clear()
    print(text)
    return

import dataloader as dl
import transformer as tr
import multitaskNet as mtn

'''
This is the main training script.
Contains the primary training loop for the Textual-LSD research network.
'''

##### Key Variables #####
EPOCHS = 1
BATCH_SIZE = 8
LR = 3e-4
USE_DOM = True
FILENAME = 'Data_8500_songs.xlsx'
ATTENTION_HEADS = 2
EMBEDDING_SIZE = 128
NUM_ENCODER_LAYERS = 1
FORWARD_XP = 2
DROPOUT = 0.25
MAXLENGTH = 200
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TRAIN_VAL_SPLIT = 0.8   

##### Load Data into Dataset#####
printc(f'Reading in {FILENAME} and creating dataloaders...')
file = pd.read_excel(FILENAME)
dataframe = pd.DataFrame(file)
dataset = dl.LSD_DataLoader(dataframe, 'lyrics', ['valence_tags', 'arousal_tags', 'dominance_tags'])
dataset.clean_lyrics(length=MAXLENGTH)
idx = np.arange(0, len(dataset), 1)
np.random.shuffle(idx)

##### Vocab work #####
printc('Creating vocabulary from training data...')
english = utils.Vocabulary(start_token='<SOS>', end_token='<EOS>', pad_token='<PAD>')
english.creat_vocabulary(np.array(dataframe['lyrics'][idx[:int(TRAIN_VAL_SPLIT * len(idx))]]), max_size=30000, min_freq=5)
PAD_IDX = english.pad_idx
VOCAB_LEN = len(english)
dataset.set_vocab(english)

##### Create Dataloaders #####
dataset_tr = Subset(dataset, idx[:int(TRAIN_VAL_SPLIT * len(idx))])
dataset_val = Subset(dataset, idx[int(TRAIN_VAL_SPLIT * len(idx)):])
dataloader_tr = DataLoader(dataset_tr, batch_size=BATCH_SIZE, shuffle=True)
dataloader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=True)

##### Prepare Model, Optimizer and Criterion #####
printc('Creating Models')
encoder = tr.Encoder(len(english), EMBEDDING_SIZE, NUM_ENCODER_LAYERS, ATTENTION_HEADS, FORWARD_XP, DROPOUT, MAXLENGTH)
print('Creating MultiTask')
multitask = mtn.multitaskNet(encoder, ATTENTION_HEADS, MAXLENGTH, EMBEDDING_SIZE, DROPOUT, USE_DOM).to(DEVICE)
print('Models Done')

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
        inp_data = torch.tensor(batch['lyrics']).to(DEVICE)
        val = torch.tensor(batch['valence_tags']).to(DEVICE)
        aro = torch.tensor(batch['arousal_tags']).to(DEVICE)
        dom = torch.tensor(batch['dominance_tags']).to(DEVICE)

        output = multitask(inp_data)

        adam.zero_grad()

        valence_loss = valence_L(output[0], val)
        arousal_loss = arousal_L(output[1], aro)
        dominance_loss = dominance_L(output[2], dom)
        loss = valence_loss + arousal_loss + dominance_loss
        loss.backward()
        torch.nn.utils.clip_grad(multitask.parameters(), max_norm=1)
        adam.step()

        losses.append(loss)
        if (batch_idx + 1) % 10:
            print(f'{batch_idx} / {len(dataloader_tr)}')
            print(loss)

torch.save(multitask.state_dict(), 'Model.pt')

print('Done')






