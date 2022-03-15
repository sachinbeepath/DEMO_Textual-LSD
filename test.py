import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

def ArgMax_to_quadrant(V, A):
    '''
    Takes in the argmaxes for valence and arousal
    1 = positive, 0 = negative  
    '''
    quads = []
    d = {'0,0':2, '0,1':1, '1,1':0, '1,0':3}
    for v, a in zip(V, A):
        a = f'{int(v)},{int(a)}'
        quads.append(d[a])
    return torch.tensor(quads)

##### Key Variables #####
BATCH_SIZE = 16
USE_DOM = True
FILENAME = 'Data_8500_songs.xlsx'
ATTENTION_HEADS = 8
EMBEDDING_SIZE = 64
NUM_ENCODER_LAYERS = 1
FORWARD_XP = 64
DROPOUT = 0.1
MAXLENGTH = 256
MT_HEADS = 8
LABEL_DICT = {'relaxed': 3, 'angry': 1, 'happy': 0, 'sad': 2}
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

PRINT_STEP = 5
SAVE_STEP = 5   
TRAIN_VAL_SPLIT = 0.8

##### Load Data into Dataset#####
printc(f'Reading in {FILENAME} and creating dataloaders...')
file = pd.read_excel(FILENAME)
dataframe = pd.DataFrame(file)
ps = PorterStemmer()
stemmer = lambda x: ps.stem(x)
dataset = dl.LSD_DataLoader(dataframe, 'lyrics', ['valence_tags', 'arousal_tags', 'dominance_tags'], MAXLENGTH)
dataset.scale_VAD_scores(5, 5)
dataset.clean_lyrics(remove_between_brackets=True, stem=True, stemmer=stemmer, tokenize=True, tokenizer=word_tokenize, length=MAXLENGTH)
idx = np.arange(0, len(dataset), 1)
np.random.shuffle(idx)

##### Vocab work #####
printc('Creating vocabulary from training data...')
english = utils.Vocabulary()
english.load('vocab_emb64.pkl')
PAD_IDX = english.pad_idx
VOCAB_LEN = len(english)
dataset.set_vocab(english)

##### Create Dataloaders #####
dataset_val = Subset(dataset, idx[int(TRAIN_VAL_SPLIT * len(idx)):])
dataloader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=True)

##### Prepare Model, Optimizer and Criterion #####
print('Creating Models') #Such a janky solution for weird difference in vocab length. Deffo gonna break
multitask = mtn.multitaskNet(MT_HEADS, MAXLENGTH+2, EMBEDDING_SIZE, DROPOUT, DEVICE, 
                            VOCAB_LEN, NUM_ENCODER_LAYERS, ATTENTION_HEADS, FORWARD_XP, 
                            PAD_IDX, USE_DOM).to(DEVICE)
multitask.double()
multitask.load_state_dict(torch.load('MTL_clasification_emb64.pt'))

losses = []
# Testing Loop
multitask.eval()
total = 0
correct_raw = 0
correct_am = 0
quad_predictions = []
predictions = []
for batch_idx, batch in enumerate(dataloader_val):
    inp_data = batch['lyrics'].to(DEVICE)
    val = batch['valence_tags'].long().to(DEVICE)
    aro = batch['arousal_tags'].long().to(DEVICE)
    dom = batch['dominance_tags'].long().to(DEVICE)
    quad = VA_to_quadrant(val, aro).to(DEVICE)
    output, quad_pred_raw = multitask(inp_data, version=0)
    quad_pred_am = ArgMax_to_quadrant(torch.argmax(output[0], dim=1), torch.argmax(output[1], dim=1)).numpy()
    quad_pred_raw = torch.argmax(quad_pred_raw, dim=1).detach().cpu().numpy()
    quad = quad.detach().cpu().numpy()
    
    if (batch_idx + 1) % PRINT_STEP == 0:
        print('')
        print(f'Batch {batch_idx + 1} / {len(dataloader_val)}')
        print(quad_pred_am, quad_pred_raw, quad)

    correct_raw += sum(quad_pred_raw == quad)
    correct_am += sum(quad_pred_am == quad)
    total += inp_data.shape[0]


print(f'Accuracy of base quadrant predictions: {100 * correct_raw / total:.4f}%')
print(f'Accuracy of VA quadrant predictions: {100 * correct_am / total:.4f}%')









