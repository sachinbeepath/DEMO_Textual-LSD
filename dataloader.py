import pandas as pd
import numpy as np
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import words as ws
import spacy 
import os

clear = lambda: os.system('cls')

def printc(x):
    clear()
    print(x)
    return
    
printc('Loading Data...')
datafile = pd.read_excel("Data_8500_songs.xlsx")
df = pd.DataFrame(datafile)

df = df[['Artist', 'song', 'valence_tags', 'arousal_tags', 'dominance_tags', 'lyrics']]

def clean_lyrics(sentence, char_to_remove = [], replacement_chars = [], stem = False, stemmer = None, 
                tokenize = False, tokenizer = None, sep = ' ', length = 100, pad_token = '<PAD>', 
                start_token = '<SOS>', end_token = '<EOS>'):
    '''
    Takes an input sentence and removes all unwanted characters, converts to lowercase, 
    and can apply stemming and tokenization.

    Parameters
    ---------------
    sentencec : String - The sentence to be converted
    char_to_remove : list<string> - List of characters to be removed from sentence
    replacement_chars : list<string> - What to replace the characters with, i.e. ' ' or ''
    stem : bool - Whether to apply stemming
    stemmer : func - Stemming function to apply
    tokenize : bool - Whether to apply tokenization
    tokenizer : func - Tokenizing function to apply
    sep : string - The string delimiter, default is is space
    length : int - Max lenght of sentences. Short sentences are padded, long sentences cropped. By default set to 100
    pad_token : string - Padding token
    start_token : string - Start of sentence token
    end_token : string - End of sentence token

    returns a numpy array of words with the selected functions applied.
    If no tokenizing is requested, the words are split by a standard string split on delimeter.
    '''
    assert len(char_to_remove) == len(replacement_chars), "Character removal list dimensions do not match"
    assert length > 0, 'Invalid sentence target length'
    assert len(sentence) > 0, 'Invalid sentence input length'

    if char_to_remove != []:
        for remove, replace in zip(char_to_remove, replacement_chars):
            sentence = sentence.replace(remove, replace)
    sentence = sentence.lower()

    if tokenize == False:
        words = sentence.split(sep)
    else:
        words = tokenizer(sentence)

    if len(words) > length:
        sentence = sentence[:length]
    
    if stem:
        words = [stemmer(word) for word in words]

    while len(words) < length:
        words.append(pad_token)
    words.insert(0, start_token)
    words.append(end_token)

    return np.array(words)

char_to_remove = ["'", '\n', '\r', ',', '!', '?', '.', '"', '_x000D_', '(', ')', '[', ']', '_', '-']
replacement_chars = ["", ' ', '', '', ' !', ' ?', ' .', '', '', '', '', '', '', '', '']
length = 500
stemmer = PorterStemmer()
tokenizer = word_tokenize

printc('Cleaning Data...')
#This step takes about 15 - 20 seconds
df['Clean_Lyrics'] = df['lyrics'].apply(lambda x: clean_lyrics(x, char_to_remove, replacement_chars, True, lambda y: stemmer.stem(y), 
                                                                True, tokenizer, length=length))
print(df['lyrics'].head())
print(df['Clean_Lyrics'][0])

print('Done')
