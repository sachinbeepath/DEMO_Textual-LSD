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
EMBEDDING_SIZE = 32 # 512
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

# train word2vec
num_features = 32
min_word_count = 5
num_workers = multiprocessing.cpu_count()
context_size = 7
downsampling = 1e-3
seed = 1

lyrics2vec = w2v.Word2Vec(sg=1, seed=seed, workers=num_workers, size=num_features, min_count=min_word_count,
    window=context_size, sample=downsampling)

lyrics2vec.build_vocab(lyrics)

lyrics2vec.train(lyrics,total_words=n, epochs =15)

if not os.path.exists("trained"):
    os.makedirs("trained")
lyrics2vec.save(os.path.join("trained", "lyrics2vec.w2v"))
lyrics2vec = w2v.Word2Vec.load(os.path.join("trained", "lyrics2vec.w2v"))

vectors = lyrics2vec.wv.vectors # embeddings of all the words in the corpus

### Getting embeddings of words in our vocab ###

# end up w tensor of dim 6.5k x 32
# building the matrix of weights to be loaded into pytorch embedding layer

weights_matrix = np.zeros(VOCAB_LEN,EMBEDDING_SIZE)
words_found = 0

for i, word in enumerate(english.vocab):
    try:
        weights_matrix[i] = lyrics2vec[word]
        words_found += 1
    except KeyError:
        weights_matrix[i] = np.random.normal(scale=0.6, size=(EMBEDDING_SIZE,))