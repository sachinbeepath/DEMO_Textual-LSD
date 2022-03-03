import pandas as pd
import numpy as np
import nltk
from nltk.corpus import words as ws
import spacy 

datafile = pd.read_excel("Data_8500_songs.xlsx")
df = pd.DataFrame(datafile)
print(df.columns)

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
    length : int - Max lenght of sentences. Short sentences are padded, long sentences are ignored. By default set to 100
    pad_token : string - Padding token
    start_token : string - Start of sentence token
    end_token : string - End of sentence token

    returns a numpy array of words with the selected functions applied.
    '''
    



