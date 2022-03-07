from operator import le
from statistics import mean
from tabnanny import verbose
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
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

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
EPOCHS = 10
BATCH_SIZE = 32
LR = 3e-4
USE_DOM = True
FILENAME = 'Data_8500_songs.xlsx'
ATTENTION_HEADS = 4
EMBEDDING_SIZE = 256
NUM_ENCODER_LAYERS = 2
FORWARD_XP = 4
DROPOUT = 0.25
MAXLENGTH = 200
MT_HEADS = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TRAIN_VAL_SPLIT = 0.8
PRINT_STEP = 25
SAVE_STEP = 5   

##### Load Data into Dataset#####
printc(f'Reading in {FILENAME} and creating dataloaders...')
file = pd.read_excel(FILENAME)
dataframe = pd.DataFrame(file)
ps = PorterStemmer()
stemmer = lambda x: ps.stem(x)
dataset = dl.LSD_DataLoader(dataframe, 'lyrics', ['valence_tags', 'arousal_tags', 'dominance_tags'], MAXLENGTH)
dataset.scale_VAD_scores(10, 5)
dataset.clean_lyrics(remove_between_brackets=True, stem=True, stemmer=stemmer, tokenize=True, tokenizer=word_tokenize, length=MAXLENGTH)
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
encoder = tr.Encoder(len(english), EMBEDDING_SIZE, NUM_ENCODER_LAYERS, ATTENTION_HEADS, FORWARD_XP, DROPOUT, MAXLENGTH+2, DEVICE).to(DEVICE)
encoder.double()
multitask = mtn.multitaskNet(encoder, MT_HEADS, MAXLENGTH+2, EMBEDDING_SIZE, DROPOUT, DEVICE, USE_DOM).to(DEVICE)
multitask.double()

adam = optim.Adam(multitask.parameters(), lr=LR) # Fine tune this hypPs...
scheduler = optim.lr_scheduler.ReduceLROnPlateau(adam, factor=0.1, patience=0.19, verbose=True)

valence_L = nn.MSELoss()
arousal_L = nn.MSELoss()
dominance_L = nn.MSELoss()

losses = []
# Training Loop
for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1} / {EPOCHS}')
    multitask.eval()
    epoch_losses = []
    for batch_idx, batch in enumerate(dataloader_tr):
        inp_data = batch['lyrics'].to(DEVICE)
        val = batch['valence_tags'].to(DEVICE)
        aro = batch['arousal_tags'].to(DEVICE)
        dom = batch['dominance_tags'].to(DEVICE)

        output = multitask(inp_data)

        adam.zero_grad()

        valence_loss = valence_L(output[0], val)
        arousal_loss = arousal_L(output[1], aro)
        dominance_loss = dominance_L(output[2], dom)
        loss = valence_loss + arousal_loss + dominance_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm(multitask.parameters(), max_norm=1)
        adam.step()
        epoch_losses.append(loss.item())

        if (batch_idx + 1) % PRINT_STEP == 0:
            print(f'Batch {batch_idx + 1} / {len(dataloader_tr)}')
            print('10 batch average loss:', np.average(epoch_losses[-10:]))
        if (batch_idx + 1) % SAVE_STEP == 0:
            losses.append(loss.item())

    mean_loss = sum(epoch_losses) / len(epoch_losses)
    scheduler.step(mean_loss)

torch.save(multitask.state_dict(), 'Model.pt')

print('Done')

plt.plot(losses)
plt.show()






