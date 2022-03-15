import numpy as np
import pickle
import pandas as pd

#def text_to_index(text, lang):
#    assert isinstance(text, np.ndarray)
#    assert isinstance(lang, Field)
#    shape = text.shape
#    text = text.flatten()
#    for i in range(len(text)):
#        text[i] = lang.vocab.stoi[text[i]]
#
#    return text.reshape(shape)

class Vocabulary():
    def __init__(self, start_token='<SOS>', end_token='<EOS>', pad_token='<PAD>'):
        '''
        start_token : string - the start of sentence indicator token
        end_token : string - the end of sentence indicator token
        pad_token : string - the sentence padding indicator token
        '''
        self.sos = start_token
        self.eos = end_token
        self.pad = pad_token

        self.vocab = None
        self.frequencies = None
        self.pad_idx = None
        self.sos_idx = None 
        self.eos_idx = None

    def creat_vocabulary(self, dataset, max_size=10000, min_freq = 5):
        '''
        dataset : array<array<string>> - an array of sentences that have been tokenized into arrays of words or characters
        max_size : int - maximum size of vocabulary (favours words that occur more frequently)
        min_freq : int - minimum number of occurences of any word to be included in the vocabulary
        '''
        dataset = np.concatenate(dataset)
        bag_of_words, counts = np.unique(dataset, return_counts=True)
        inds = np.argsort(counts)[::-1]
        bag_of_words = bag_of_words[inds]
        counts = counts[inds]

        missing_tokens = []

        for token in [self.sos, self.eos, self.pad]:
            if token not in bag_of_words:
                missing_tokens.append(token)

        bag_of_words = bag_of_words[counts >= min_freq]
        counts = counts[counts >= min_freq]

        if len(bag_of_words) > max_size - len(missing_tokens):
            bag_of_words = bag_of_words[:max_size - len(missing_tokens)]
            counts = counts[:max_size - len(missing_tokens)]
        
        for tok in missing_tokens:
            bag_of_words = np.append(bag_of_words, tok)
            counts = np.append(counts, 1)

        self.vocab = bag_of_words
        self.frequencies = counts

        self.pad_idx = np.where(self.pad == self.vocab)[0][0]
        self.sos_idx = np.where(self.sos == self.vocab)[0][0]
        self.eos_idx = np.where(self.eos == self.vocab)[0][0]
        return

    def __len__(self):
        if self.vocab is not None:
            return len(self.vocab)
        else:
            return 0

    def __getitem__(self, idx):
        if self.vocab is not None:
            return self.vocab[idx]
        else:
            return None

    def str_to_ind(self, arr):
        '''
        Converts an array of strings into an array of indices from the current vocab
        '''
        if self.vocab is None:
            return None

        get_ind = lambda word: np.where(self.vocab == word)[0][0] if word in self.vocab else self.pad_idx
        get_ind = np.vectorize(get_ind)

        return get_ind(arr)

    def save(self, filename):
        savefile = [self.sos, self.eos, self.pad, self.sos_idx, self.eos_idx, self.pad_idx, self.vocab, self.frequencies]
        pickle.dump(savefile, open(filename, 'wb'))
        return
    
    def load(self, file):
        loadfile = pickle.load(open(file, 'rb'))
        self.sos, self.eos, self.pad, self.sos_idx, self.eos_idx, self.pad_idx, self.vocab, self.frequencies = loadfile
        return


def generate_test_val(dataframe, split, fnames, type='excel'):
    '''
    Creates seperate files for train and val sets.

    Parameters
    ---------------------
    dataframe : Pandas.DataFrame - dataframe to be split
    split : float 0 < s < 1 - proportion to be assigned to train
    fnames : list<string> - name for each train and val file, inluding extension
    type : string - type of save format (excel, csv, pickle)
    '''
    assert isinstance(dataframe, pd.DataFrame), 'Not a dataframe!'
    assert type in ['excel', 'csv', 'pickle'], 'invalid save format!'
    assert len(fnames) == 2, 'Must be exactly two file names'
    assert split < 1, 'Split must be less than 1'
    assert split > 0, 'Split must be greater than 0'

    idx = np.arange(0, len(dataframe), 1)
    np.random.shuffle(idx)
    tr = idx[:int(len(idx) * split)]
    val = idx[int(len(idx) * split):]

    train = dataframe.iloc[tr, :]
    validation = dataframe.iloc[val, :]
    if type == 'excel':
        train.to_excel(fnames[0])
        validation.to_excel(fnames[1])
    elif type == 'csv':
        train.to_csv(fnames[0])
        validation.to_csv(fnames[1])
    if type == 'pickle':
        train.to_pickle(fnames[0])
        validation.to_pickle(fnames[1])
    print('Files Saved')
    return

"""
FILENAME = 'Data_8500_songs.xlsx'
file = pd.read_excel(FILENAME)
dataframe = pd.DataFrame(file)

generate_test_val(dataframe, 0.8, ['8500_songs_training.xlsx', '8500_songs_validation.xlsx'])
"""
