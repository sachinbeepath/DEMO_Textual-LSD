import torch
import utils
import torch.nn as nn
import numpy as np
from gensim.models import Word2Vec
import multiprocessing
import pickle
import bcolz


##### Key Variables #####
# Hashed values are those used in the reference paper
EPOCHS = 1 #Until convergence
BATCH_SIZE = 32 # 8
FILENAME = 'Datasets/train_balanced.xlsx'
MAXLENGTH = 1024
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using ', DEVICE)

trainer = utils.Textual_LSD_TVT(verbose=True)

trainer.load_dataset(FILENAME, MAXLENGTH, BATCH_SIZE)
trainer.load_vocab('Pickles/NewVocab.pkl')

vocab_words = trainer.vocab.vocab
VOCAB_LEN = len(vocab_words)

words = []
idx = 0
word2idx = {}

vectors = bcolz.carray(np.zeros(1), rootdir=f'Datasets/glove6B.50.txt', mode='w')

with open(f'Datasets/glove.6B.50d.txt', 'rb') as f:
    for l in f:
        line = l.decode().split()
        word = line[0]
        words.append(word)
        word2idx[word] = idx
        idx += 1
        vect = np.array(line[1:]).astype(float)
        vectors.append(vect)

vectors = bcolz.carray(vectors[1:].reshape((400001, 50)), rootdir=f'Datasets/glove6B.50.txt', mode='w')
vectors.flush()
pickle.dump(words, open(f'../glove6B.50_words.pkl', 'wb'))
pickle.dump(word2idx, open(f'../glove6B.50_idx.pkl', 'wb'))

vectors = bcolz.open(f'Datasets/glove6B.50.txt')[:]
words = pickle.load(open(f'../glove6B.50_words.pkl', 'rb'))
word2idx = pickle.load(open(f'../glove6B.50_idx.pkl', 'rb'))


glove = {w: vectors[word2idx[w]] for w in words}

weights_matrix = np.zeros((VOCAB_LEN, 50))
words_found = 0

for i, word in enumerate(vocab_words):
    try:
        weights_matrix[i] = glove[word]
        words_found += 1
    except KeyError:
        weights_matrix[i] = np.random.normal(scale=0.6, size=(1,50))

w2v_weights_matrix = torch.tensor(weights_matrix)
torch.save(w2v_weights_matrix, 'glove_weights_1000.pkl',pickle_module = pickle)