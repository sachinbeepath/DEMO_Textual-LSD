import numpy as np
import pickle
import time
import pandas as pd
import torch
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import dataloader as dl
import multitaskNet as mtn
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import multitaskNet as mtn
import matplotlib.pyplot as plt

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

class Textual_LSD_Training():
    def __init__(self, verbose=False):
        self.verbose = verbose

        self.dataset = None
        self.dataloader = None
        self.dataframe = None
        self.stemmer = None
        self.tokenizer = None
        self.max_length = None

        self.vocab = None
        self.pad_idx = None
        self.vocab_len = None

        self.multitask = None
        self.optim = None
        self.scheduler = None
        self.valence_L = None
        self.arousal_L = None
        self.dominance_L = None
        self.quad_L = None
        self.device = None
        self.use_dom = None

        self.losses = []
        self.valpoints = []
        self.aropoints = []
        self.true_quads = []
        self.accuracy = []

    def load_dataset(self, fname, max_length, batch_size, lyric_col='lyrics', 
                        label_cols = ['valence_tags', 'arousal_tags', 'dominance_tags'], 
                        remove_between_brackets=True, stem=True, stemmer=PorterStemmer(), 
                        tokenize=True, tokenizer=word_tokenize):
        if self.verbose:
            print('Starting Load Dataset...')
        file = pd.read_excel(fname)
        df = pd.DataFrame(file)
        ps = stemmer
        self.stemmer = lambda x: ps.stem(x)
        dataset = dl.LSD_DataLoader(df, lyric_col, label_cols, max_length)
        dataset.scale_VAD_scores(5, 5)
        dataset.clean_lyrics(remove_between_brackets=remove_between_brackets, stem=stem, stemmer=self.stemmer, 
                                tokenize=tokenize, tokenizer=tokenizer, length=max_length)
        dataloader_tr = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.dataset = dataset
        self.dataloader = dataloader_tr
        self.dataframe = df
        self.max_length = max_length
        if self.verbose:
            print("Succesfully Loaded Dataframe")
        return

    def generate_vocab(self, save=False, save_name=None):
        if self.verbose:
            print('Starting Generate Vocab...')
        assert self.dataset is not None, 'Please load in a dataset before generating a vocabulary'
        assert self.dataframe is not None, 'Please load in a dataframe before generating a vocabulary'

        self.vocab = Vocabulary(start_token='<SOS>', end_token='<EOS>', pad_token='<PAD>')
        self.vocab.creat_vocabulary(np.array(self.dataframe['lyrics']), max_size=30000, min_freq=5)
        self.pad_idx = self.vocab.pad_idx
        self.vocab_len = len(self.vocab)
        self.dataset.set_vocab(self.vocab)
        if save:
            assert save_name != None, 'Please provide a save name for the file!'
            self.vocab.save(save_name)
            print(f'Vocabulary Saved to {save_name}')
        if self.verbose:
            print('Successfully generated vocabulary')
            print(f'Length: {self.vocab_len}, Padding index: {self.pad_idx}')
        return 
    
    def load_vocab(self, fname):
        if self.verbose:
            print('Starting Load Vocab...')
        assert self.dataset is not None, 'Please load in a dataset before loading a vocabulary'

        self.vocab = Vocabulary()
        self.vocab.load(fname)
        self.dataset.set_vocab(self.vocab)
        self.vocab_len = len(self.vocab)
        self.pad_idx = self.vocab.pad_idx
        if self.verbose:
            print(f'Successfully Loaded {fname} into vocabulary')
        return

    def generate_models(self, emb_size, att_heads, drp, dom, lr, mt_heads, num_enc, 
                        forw_exp, dev, lr_fact=0.2, lr_pat=10, lr_verbose=True, w2v=None):
        if self.verbose:
            print('Starting Generate Models...')
        assert self.vocab_len is not None, 'Please generate or load a vocabulary before training'
        assert self.pad_idx is not None, 'Please generate or load a vocabulary before training'

        self.multitask = mtn.multitaskNet(mt_heads, self.max_length+2, emb_size, drp, dev, 
                                self.vocab_len, num_enc, att_heads, forw_exp, 
                                self.pad_idx, dom, w2v).to(dev)
        self.multitask.double()

        self.optim = optim.AdamW(self.multitask.parameters(), lr=lr) # Fine tune this hypPs...
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optim, factor=lr_fact, patience=lr_pat, verbose=lr_verbose)

        self.valence_L = nn.CrossEntropyLoss()
        self.arousal_L = nn.CrossEntropyLoss()
        self.dominance_L = nn.CrossEntropyLoss()
        self.quad_L = nn.CrossEntropyLoss()
        self.device = dev
        self.use_dom = dom
        if self.verbose:
            print('Models Generated...')
        return 

    def load_models(self, fname, lr, lr_fact=0.2, lr_pat=10, lr_verbose=True):
        if self.verbose:
            print('Starting Load Models...')
        self.multitask.load_state_dict(torch.load(fname, map_location=self.device))
        self.optim = optim.AdamW(self.multitask.parameters(), lr=lr) # Fine tune this hypPs...
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optim, factor=lr_fact, patience=lr_pat, verbose=lr_verbose)
        if self.verbose:
            print('Models Loaded')
        return

    def VA_to_quadrant(self, V, A):
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

    def train(self, epochs, print_step, save_step, save_name=None, 
                show_preds=False, show_acc=True, show_loss=True, show_time=True, enc_version=1):
        if self.verbose:
            print(f'Number of batches per epoch: {len(self.dataloader)}')
            print(f'Printing every {print_step} batches, saving every {save_step} batches')

        # Training loop
        self.multitask.train()
        for epoch in range(epochs):
            total = 0
            correct = 0
            print(f'Epoch {epoch + 1} / {epochs}')
            t_0 = time.time()
            t = time.time()
            epoch_l = []
            for batch_idx, batch in enumerate(self.dataloader):
                inp_data = batch['lyrics'].to(self.device)
                val = batch['valence_tags'].long().to(self.device)
                aro = batch['arousal_tags'].long().to(self.device)
                dom = batch['dominance_tags'].long().to(self.device)
                quad = self.VA_to_quadrant(val, aro).to(self.device)
                output, quad_pred = self.multitask(inp_data, version=enc_version)
                self.optim.zero_grad()
                
                valence_loss = self.valence_L(output[0], val)
                arousal_loss = self.arousal_L(output[1], aro)
                dominance_loss = self.dominance_L(output[2], dom) if self.use_dom == True else torch.tensor([0])
                quad_loss = self.quad_L(quad_pred, quad)
                loss = arousal_loss + valence_loss + dominance_loss + quad_loss
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(multitask.parameters(), max_norm=1)
                self.optim.step()
                epoch_l.append(loss.item())
                # Calcuate Accuracy
                total += len(batch)
                for i in range(len(batch)):
                    correct += 1 if np.argmax(quad_pred[i].detach().cpu().numpy()) == quad[i] else 0

                if (batch_idx + 1) % print_step == 0:
                    if self.verbose:
                        print('')
                        print(f'Batch {batch_idx + 1} / {len(self.dataloader)}')
                    if show_preds:
                        print('VAL pred/true', output[0, :2].detach().cpu().numpy(), val[:2].detach().cpu().numpy())
                        print('ARO pred/true', output[1, :2].detach().cpu().numpy(), aro[:2].detach().cpu().numpy())
                        print('DOM pred/true', output[2, :2].detach().cpu().numpy(), dom[:2].detach().cpu().numpy())
                        print('Quadrant pred/true', np.argmax(quad_pred.detach().cpu().numpy(), axis=1), quad.detach().cpu().numpy())
                    if show_loss:
                        print(f'{print_step} batch average loss:', np.average(epoch_l[-print_step:]))

                if (batch_idx + 1) % save_step == 0:
                    self.losses.append(np.average(epoch_l[-save_step:]))
                    if batch_idx != len(self.dataloader) and (epoch + 1) % 8 == 0 : #ignore final batch since differnt size
                        self.valpoints.append(torch.squeeze(torch.softmax(output, dim=2)[0, :, 0]).detach().cpu().numpy())
                        self.aropoints.append(torch.squeeze(torch.softmax(output, dim=2)[1, :, 0]).detach().cpu().numpy())
                    self.accuracy.append(correct / total)
                    total, correct = 0, 0
                    self.scheduler.step(self.losses[-1])
            if show_time:
                print(f'Epoch Time: {time.time() - t:.1f}s')
        
        # Trainig Loop Complete
        if save_name != None:
            torch.save(self.multitask.state_dict(), save_name)
            if self.verbose:
                print(f'Successfully saved model weights as {save_name}')
        else:
            print('You have not entered a save name for the model')
            name = input('If you wish to save it, please type a file name now (.pt), otherwise enter N: ')
            if name not in ['n', 'N', ' n', ' N', 'no', 'No']:
                torch.save(self.multitask.state_dict(), name)
                if self.verbose:
                    print(f'Successfully saved model weights as {name}')
            else:
                print(f'Training Compeleted. Total time: {time.time() - t_0:.0f}s')
        return 
        
    def return_values(self, losses=True, acc=False, val_preds=False, aro_preds=False):
        returns = []
        if losses:
            returns.append(self.losses)
        if acc:
            returns.append(self.accuracy)
        if val_preds:
            returns.append(self.valpoints)
        if aro_preds:
            returns.append(self.aropoints)
        return tuple(returns)

    def plot_data(self, averaging_window=20):
        w = np.ones(averaging_window) / averaging_window
        fig, axs = plt.subplots(2)
        axs[0].plot(np.convolve(self.losses[averaging_window:-averaging_window], w))
        axs[0].set_title('Training Losses')
        axs[1].plot(np.convolve(self.accuracy[averaging_window:-averaging_window], w))
        axs[1].set_title('Quadrant Prediction Accuracy')
        plt.show()
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

