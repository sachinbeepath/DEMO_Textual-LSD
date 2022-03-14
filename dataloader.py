import pandas as pd
import numpy as np
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import words as ws 
import os
import re
from lyricsScraper import LyricScraper
import torch
from torch.utils.data import Dataset
import utils

'''
--------LSD_DataLoader-----------

A custom pytorch dataset class.
It is used to take in a dataset of song information, with minimum of one column
containing song lyrics. It can then interface with a Pytorch Dataloader class through
the __len__ and __getitem__ methods. It also as additional functionality in its methods 
for cleaning lyrics, tokenizing sentences and applying stemming algorithms for 
NLP preperation.
'''

class LSD_DataLoader(Dataset):
    def __init__(self, dataFrame,  lyric_col, label_cols, length, lyric_format = 'string', label_type='float', label_dict=None):
        '''
        DataSet class for a pandas dataframe containing song lyrics.

        Parameters:
        -------------------
        dataFrame : pandas.DataFrame - dataframe object containing at least a lyrics colunn
        lyric_col : string - column title for the lyrics column
        label_cols : list<string> - list of column titles referring to V, A and D values.
        lyric_format : string - one of "string" or "list". This determines if the lyrics are stored as one continuous string
                        or as a list of strings with some delimiter character
                        If set to "list" then character removal, remove_between_brackets and tokenization are automatically skipped
        label_type : string - the datatype of the labels. Affects how they are converted to tensor. ('float', 'string')
        label_dict : dict - optional dictionary for converting labels to different kind. Required for 'string' label type
        '''
        super().__init__()
        assert isinstance(dataFrame, pd.DataFrame), 'dataFrame must be a Pandas DataFrame object.'
        assert lyric_format in ['string', 'list'], 'lyric_format must be in ["string", "list"]'

        self.original_df = dataFrame # Kept uneditted in case needed later
        self.df = dataFrame # May be editted my methods within the class
        self.columns = dataFrame.columns # List of column headers
        self.lyric_col = lyric_col
        self.label_cols = label_cols
        self.format = lyric_format
        self.length = length # make this actually useful
        self.vocab = None
        self.label_type = label_type
        self.label_dict = label_dict
        
        #Default lists for character replacement/removal
        self.char_to_remove_default = ["'", '\n', '\r', ',', '!', '?', '.', '"', '_x000D_', '(', ')', '[', ']', '_', '-', '*', '%']
        self.replacement_chars_default = ["", ' ', '', '', ' !', ' ?', ' .', '', '', '', '', '', '', '', '', '', '']

    def __len__(self):
        return len(self.df[self.columns[0]])
    
    def __getitem__(self, idx):
        cols = np.array(self.label_cols)
        if self.label_type == 'float':
            out = {col : torch.tensor((self.df[col][idx] < 0) * 1) for col in cols}
        elif self.label_type == 'string':
            out = {col : torch.tensor(self.df[col].apply(lambda x: self.label_dict[x])[idx]) for col in cols}
        out[self.lyric_col] = torch.tensor(self.vocab.str_to_ind(self.df[self.lyric_col][idx]))
        return out

    def set_vocab(self, v):
        self.vocab = v
        return

    def scale_VAD_scores(self, scale, mean):
        for col in self.label_cols:
            self.df[col] = self.df[col].apply(lambda x: (x - mean) / scale)
        return

    def change_lyric_format(self, delimiter=' '):
        '''
        Changes the formatting of lyric column from one string to an array of strings seperated by a 
        delimiting character, and vice versa, with array elements being joined with the delimiting
        character.

        delimiter : string - the character used to seperate or join words
        '''
        if self.format == 'string':
            self.df[self.lyric_col].apply(lambda s: s.split(delimiter))
            self.format = 'list'
            return
        elif self.format == 'list':
            self.df[self.lyric_col].apply(lambda s: delimiter.join(s))
            self.format = 'string'
            return

    def restore_default_dataframe(self):
        '''Sets editable dataframe back to original dataframe (as passed in during initialisationg)'''
        self.df = self.original_df
        return

    def get_dataframe(self, original=False):
        '''
        Retrieves the dataframe. By default, the editted df is returned. If original is true then the
        unedited version (the one passed in at initialisation) is returned
        '''

        if original:
            return self.original_df
        else:
            return self.df

    def clean_lyrics(self, char_to_remove = 'Default', replacement_chars = 'Default', remove_between_brackets=False, stem = False, stemmer = None, 
                    tokenize = False, tokenizer = None, sep = ' ', length = 500, pad_token = '<PAD>', 
                    start_token = '<SOS>', end_token = '<EOS>'):
        '''
        Takes the lyrics from the dataframe lyrics column and removes all unwanted characters, converts to lowercase, 
        and can apply stemming and tokenization.

        Parameters
        ---------------
        char_to_remove : list<string> - List of characters to be removed from sentence. If 'Default', applies default list
        replacement_chars : list<string> - What to replace the characters with, i.e. ' ' or ''. If 'Default', applies default list
        remove_between_brackets : bool - whether to remove all text between brackets, e.g. (Ooooh, ahhhh), [instrumental section]
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

        if char_to_remove == 'Default':
            char_to_remove = self.char_to_remove_default
        if replacement_chars == 'Default':
            replacement_chars = self.replacement_chars_default

        self.df[self.lyric_col] = self.df[self.lyric_col].apply(lambda song: self.__clean(song, char_to_remove, replacement_chars, remove_between_brackets, 
                                                stem, stemmer, tokenize, tokenizer, sep, length, pad_token, 
                                                start_token, end_token))
        self.format = 'list'
        return

    def __clean(self, sentence, char_to_remove, replacement_chars, remove_between_brackets, stem, stemmer, 
                    tokenize, tokenizer, sep, length, pad_token, start_token, end_token):
        # All parameters are the same as self.clean_lyrics, except sentence. This is a string containing the lyrics for one song.
        
        assert len(char_to_remove) == len(replacement_chars), "Character removal list dimensions do not match"
        assert length > 0, 'Invalid sentence target length'
        assert len(sentence) > 0, 'Invalid sentence input length'

        if self.format == 'string': # Only do character removal and tokenization if lyrics are one string. 
            if remove_between_brackets:
                re.sub("[\(\[\{].*?[\)\]\}]", "", sentence)

            if char_to_remove != []:
                for remove, replace in zip(char_to_remove, replacement_chars):
                    sentence = sentence.replace(remove, replace)
            sentence = sentence.lower()

            if tokenize == False:
                words = sentence.split(sep)
            else:
                words = tokenizer(sentence)
        else:
            words = sentence 

        if len(words) > length:
            words = words[:length]
        
        if stem:
            words = [stemmer(word) for word in words]

        while len(words) < length:
            words.append(pad_token)
        words.insert(0, start_token)
        words.append(end_token)
        return np.array(words)









