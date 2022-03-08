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
This is the main testing script.
Contains the primary testing loop for the Textual-LSD research network.
'''
def VA_to_quadrant(v, a):
    if v > 0:
        if a > 0:
            return 1
        else:
            return 4
    else:
        if a > 0:
            return 2
        else:
            return 3


##### Key Variables #####
BATCH_SIZE = 8
USE_DOM = True
FILENAME = 'Comparison_2500_songs_lyrics.xlsx'
ATTENTION_HEADS = 8
EMBEDDING_SIZE = 256
NUM_ENCODER_LAYERS = 3
FORWARD_XP = 4
DROPOUT = 0.25
MAXLENGTH = 500
MT_HEADS = 4
LABEL_DICT = {'relaxed': 4, 'angry': 2, 'happy': 1, 'sad': 3}
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

PRINT_STEP = 5
SAVE_STEP = 5   

##### Load Data into Dataset#####
printc(f'Reading in {FILENAME} and creating dataloaders...')
file = pd.read_excel(FILENAME)
dataframe = pd.DataFrame(file)
ps = PorterStemmer()
stemmer = lambda x: ps.stem(x)
dataset = dl.LSD_DataLoader(dataframe, 'Lyrics', ['Mood'], MAXLENGTH, 'string', 'string', LABEL_DICT)
dataset.clean_lyrics(remove_between_brackets=True, stem=True, stemmer=stemmer, tokenize=True, tokenizer=word_tokenize, length=MAXLENGTH)
idx = np.arange(0, len(dataset), 1)
np.random.shuffle(idx)

##### Vocab work #####
printc('Creating vocabulary from training data...')
english = utils.Vocabulary()
english.load('vocab_big.pkl')
PAD_IDX = english.pad_idx
VOCAB_LEN = len(english)
dataset.set_vocab(english)

##### Create Dataloaders #####
dataloader_te = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

##### Prepare Model, Optimizer and Criterion #####
print('Creating Models') #Such a janky solution for weird difference in vocab length. Deffo gonna break
encoder = tr.Encoder(VOCAB_LEN, EMBEDDING_SIZE, NUM_ENCODER_LAYERS, ATTENTION_HEADS, FORWARD_XP, DROPOUT, MAXLENGTH+2, DEVICE).to(DEVICE)
encoder.double()
encoder.load_state_dict(torch.load('ENC_big.pt'))
multitask = mtn.multitaskNet(encoder, MT_HEADS, MAXLENGTH+2, EMBEDDING_SIZE, DROPOUT, DEVICE, USE_DOM).to(DEVICE)
multitask.double()
multitask.load_state_dict(torch.load('MTL_big.pt'))

losses = []
# Testing Loop
multitask.eval()
encoder.eval()
total = 0
correct = 0
quad_predictions = []
predictions = []
for batch_idx, batch in enumerate(dataloader_te):
    inp_data = batch['Lyrics'].to(DEVICE)
    quadrant = batch['Mood'].to(DEVICE)

    output = torch.flatten(multitask(inp_data), start_dim=1)
    print(output.shape)
    print(quadrant.shape)

    for i in range(output.shape[1]):
        pred = VA_to_quadrant(output[0, i].item(), output[1, i].item())
        if pred == quadrant[i]:
            correct += 1
        total += 1
        quad_predictions.append(pred)
        predictions.append(output[:, i].cpu().detach().numpy())

    if (batch_idx + 1) % PRINT_STEP == 0:
        print(f'Batch {batch_idx + 1} / {len(dataloader_te)}')

print(f'Accuracy of quadrant predictions: {100 * correct / total:.4f}%')

torch.save(multitask.state_dict(), 'Model.pt')

fig, axs = plt.subplots(nrows=2, ncols=2)
axs[0,0].plot(losses)
axs[0,1].plot()
#axs[0,1].plot()
plt.show()





