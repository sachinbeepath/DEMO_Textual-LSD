from tkinter import Y
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

def VA_to_quadrant(V, A):
    quads = []
    for v, a in zip(V, A):
        if v > 0:
            if a > 0:
                quads.append(0)
            else:
                quads.append(3)
        else:
            if a > 0:
                quads.append(1)
            else:
                quads.append(2)
    return torch.tensor(quads)

##### Key Variables #####
# Hashed values are those used in the reference paper
EPOCHS = 40 #Until convergence
BATCH_SIZE = 16 # 8
LR = 3e-4 #2e-5
USE_DOM = True
FILENAME = '8500_songs_training.xlsx'
ATTENTION_HEADS = 8 # 8
EMBEDDING_SIZE = 64 # 512
NUM_ENCODER_LAYERS = 1 # 3
FORWARD_XP = 64
DROPOUT = 0.1 # 0.1
MAXLENGTH = 256 #1024
MT_HEADS = 8 # 8
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using ', DEVICE)

PRINT_STEP = 200
SAVE_STEP = 10  

##### Load Data into Dataset#####
printc(f'Reading in {FILENAME} and creating dataloaders...')
file = pd.read_excel(FILENAME)
dataframe = pd.DataFrame(file)
ps = PorterStemmer()
stemmer = lambda x: ps.stem(x)
dataset = dl.LSD_DataLoader(dataframe, 'lyrics', ['valence_tags', 'arousal_tags', 'dominance_tags'], MAXLENGTH)
dataset.scale_VAD_scores(5, 5)
dataset.clean_lyrics(remove_between_brackets=True, stem=True, stemmer=stemmer, tokenize=True, tokenizer=word_tokenize, length=MAXLENGTH)
dataloader_tr = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

##### Vocab work #####
printc('Creating vocabulary from training data...')
english = utils.Vocabulary(start_token='<SOS>', end_token='<EOS>', pad_token='<PAD>')
english.creat_vocabulary(np.array(dataframe['lyrics']), max_size=30000, min_freq=5)
PAD_IDX = english.pad_idx
VOCAB_LEN = len(english)
dataset.set_vocab(english)
print(VOCAB_LEN)
english.save('vocab_emb64.pkl')

##### Prepare Model, Optimizer and Criterion #####
print('Creating Models')
multitask = mtn.multitaskNet(MT_HEADS, MAXLENGTH+2, EMBEDDING_SIZE, DROPOUT, DEVICE, 
                            VOCAB_LEN, NUM_ENCODER_LAYERS, ATTENTION_HEADS, FORWARD_XP, 
                            PAD_IDX, USE_DOM).to(DEVICE)
multitask.double()

adam = optim.AdamW(multitask.parameters(), lr=LR) # Fine tune this hypPs...
scheduler = optim.lr_scheduler.ReduceLROnPlateau(adam, factor=0.2, patience=10, verbose=True)

valence_L = nn.CrossEntropyLoss()
arousal_L = nn.CrossEntropyLoss()
dominance_L = nn.CrossEntropyLoss()
quad_L = nn.CrossEntropyLoss()

losses = []
valpoints = []
aropoints = []
true_quads = []
multitask.train()
# Training Loop
for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1} / {EPOCHS}')
    epoch_losses = []
    t = time.time()
    for batch_idx, batch in enumerate(dataloader_tr):
        inp_data = batch['lyrics'].to(DEVICE)
        val = batch['valence_tags'].long().to(DEVICE)
        aro = batch['arousal_tags'].long().to(DEVICE)
        dom = batch['dominance_tags'].long().to(DEVICE)
        quad = VA_to_quadrant(val, aro).to(DEVICE)
        output, quad_pred = multitask(inp_data, version=0)
        adam.zero_grad()

        valence_loss = valence_L(output[0], val)
        arousal_loss = arousal_L(output[1], aro)
        dominance_loss = dominance_L(output[2], dom) if USE_DOM == True else torch.tensor([0])
        quad_loss = quad_L(quad_pred, quad)
        loss = arousal_loss + valence_loss + dominance_loss + quad_loss
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(multitask.parameters(), max_norm=1)
        adam.step()
        epoch_losses.append(loss.item())
        if batch_idx != len(dataloader_tr) and (epoch + 1) % 8 == 0 : #ignore final batch since differnt size
            valpoints.append(torch.squeeze(torch.softmax(output, dim=2)[0, :, 0]).detach().cpu().numpy())
            aropoints.append(torch.squeeze(torch.softmax(output, dim=2)[1, :, 0]).detach().cpu().numpy())


        if (batch_idx + 1) % PRINT_STEP == 0:
            print('')
            #print(quad, val, aro)
            #print(f'True Quadrant: {quad.detach().cpu().numpy()}, Predicted: {np.argmax(quad_pred.detach().cpu().numpy(), axis=1)}')
            #print(f'Quadrant Scores: {quad_pred}')
            print(f'Batch {batch_idx + 1} / {len(dataloader_tr)}')
            #print('VAL', output[0, :2].detach().cpu().numpy(), val[:2].detach().cpu().numpy())
            #print('ARO', output[1, :2].detach().cpu().numpy(), aro[:2].detach().cpu().numpy())
            #print('DOM', output[2, :2].detach().cpu().numpy(), dom[:2].detach().cpu().numpy())
            print(f'{PRINT_STEP} batch average loss:', np.average(epoch_losses[-PRINT_STEP:]))
            scheduler.step(np.average(epoch_losses[-PRINT_STEP:]))
        if (batch_idx + 1) % SAVE_STEP == 0:
            losses.append(np.average(epoch_losses[-SAVE_STEP:]))
    print(f'Epoch Time: {time.time() - t:.1f}s')
    mean_loss = sum(epoch_losses) / len(epoch_losses)

torch.save(multitask.state_dict(), 'MTL_clasification_1epoch.pt')

valpoints = np.concatenate(valpoints).reshape(-1)
aropoints = np.concatenate(aropoints).reshape(-1)
print(valpoints)

print('Done')

plt.scatter(valpoints, aropoints, s=0.2, c=np.repeat(np.linspace(0, 1, int(len(valpoints) / BATCH_SIZE)), BATCH_SIZE))
plt.legend()
plt.show()






