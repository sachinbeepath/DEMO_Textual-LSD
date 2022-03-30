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
import nltk
import gensim.models.word2vec as w2v
import multiprocessing
import logging
import codecs
import bcolz
import pickle

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
EMBEDDING_SIZE = 50 # 512
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
dataset.get_dataframe()

##### Vocab work #####
print('Creating vocabulary from training data...')
english = utils.Vocabulary(start_token='<SOS>', end_token='<EOS>', pad_token='<PAD>')
english.load('vocab_emb64.pkl')
PAD_IDX = english.pad_idx
VOCAB_LEN = len(english)
dataset.set_vocab(english)
print(VOCAB_LEN)


#### WORD2VEC ####
corpus = dataset.get_dataframe()
lyrics = corpus.lyrics

# changing lyrics to a list
n = len(lyrics)
main_list = []
for i in range(n):
    sub_list=[]
    for word in lyrics[i]:
        sub_list.append(word)
    main_list.append(sub_list)
lyrics = main_list

words = []
idx = 0
word2idx = {}

vectors = bcolz.carray(np.zeros(1), rootdir=f'glove6B.50.txt', mode='w')

with open(f'glove.6B.50d.txt', 'rb') as f:
    for l in f:
        line = l.decode().split()
        word = line[0]
        words.append(word)
        word2idx[word] = idx
        idx += 1
        vect = np.array(line[1:]).astype(float)
        vectors.append(vect)

vectors = bcolz.carray(vectors[1:].reshape((400001, 50)), rootdir=f'glove6B.50.txt', mode='w')
vectors.flush()
pickle.dump(words, open(f'glove6B.50_words.pkl', 'wb'))
pickle.dump(word2idx, open(f'glove6B.50_idx.pkl', 'wb'))

vectors = bcolz.open(f'glove6B.50.txt')[:]
words = pickle.load(open(f'glove6B.50_words.pkl', 'rb'))
word2idx = pickle.load(open(f'glove6B.50_idx.pkl', 'rb'))


glove = {w: vectors[word2idx[w]] for w in words}

matrix_len = len(lyrics)
weights_matrix = np.zeros((matrix_len, 50))
words_found = 0

for i, word in enumerate(english.vocab):
    try:
        weights_matrix[i] = glove[word]
        words_found += 1
    except KeyError:
        weights_matrix[i] = np.random.normal(scale=0.6, size=(1,EMBEDDING_SIZE ))

w2v_weights_matrix = torch.tensor(weights_matrix)
torch.save(w2v_weights_matrix, 'glove_weights.pkl',pickle_module= pickle)