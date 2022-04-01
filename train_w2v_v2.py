import torch
import utils
import torch.nn as nn
import numpy as np
from gensim.models import Word2Vec
import multiprocessing
import pickle


##### Key Variables #####
# Hashed values are those used in the reference paper
EPOCHS = 1 #Until convergence
BATCH_SIZE = 32 # 8
FILENAME = 'Datasets/train_balanced.xlsx'
EMBEDDING_SIZE = 32 # 512
MAXLENGTH = 1024
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using ', DEVICE)

trainer = utils.Textual_LSD_TVT(verbose=True)

trainer.load_dataset(FILENAME, MAXLENGTH, BATCH_SIZE)
trainer.load_vocab('Pickles/Balanced_Vocab.pkl')

corpus = trainer.dataframe
lyrics = corpus.lyrics

# fix formatting
n = len(lyrics)
main_list = []
for i in range(n):
    sub_list=[]
    for word in lyrics[i]:
        sub_list.append(word)
    main_list.append(sub_list)
lyrics = main_list

words = trainer.vocab.vocab
VOCAB_LEN = len(words)
# words = np.unique(words) # making sure all words are unique

# build w2v model
cores = multiprocessing.cpu_count() # Count the number of cores in a computer
w2v_model = Word2Vec(min_count=5,
                     window=2,
                     size=EMBEDDING_SIZE,
                     sample=6e-5,
                     alpha=0.03,
                     min_alpha=0.0007,
                     negative=20,
                     workers=cores-1)

w2v_model.build_vocab(lyrics, progress_per=10000)

w2v_model.train(lyrics, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)

weights_matrix = np.zeros((VOCAB_LEN, EMBEDDING_SIZE))
words_found = 0

for i, word in enumerate(words):
    try:
        weights_matrix[i] = w2v_model.wv.__getitem__(word)
        words_found += 1
    except KeyError:
        weights_matrix[i] = np.random.normal(scale=0.6, size=(1,EMBEDDING_SIZE))

weights_matrix = torch.tensor(weights_matrix, requires_grad=True)
torch.save(weights_matrix, 'w2v_weights.pkl',pickle_module= pickle)