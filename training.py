import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import argparse

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

##### Prepare Model, Optimizer and Criterion #####
encoder = None
multitask = mtn.multitaskNet(encoder, ATTENTION_HEADS, dataset.length, EMBEDDING_SIZE, DROPOUT, USE_DOM).to(device=DEVICE)

adam = optim.Adam(multitask.parameters(), lr=LR) # Fine tune this hypPs...

valence_L = nn.MSELoss()
arousal_L = nn.MSELoss()
dominance_L = nn.MSELoss()

# Training Loop




