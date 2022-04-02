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
#from imblearn.over_sampling import RandomOverSampler
import os
from sklearn.metrics import confusion_matrix

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

class Textual_LSD_TVT():
    '''
    The Textual_LSD_TVT class allows for concise implementation of trainig, validation, 
    and testing routines. In houses self-contained methods for loading in .xlsx datasets, 
    creating and loading in vocabularies (utilising the custom Vocabulary class also in this utils file),
    generating and loading in Textual_LSD models, as well as performing Training, Testing and Validation.

    There are additional methods allowing for the retrieval of training and testing data such 
    as accuracy, losses, precision, recall and F-scores etc., as well as plotting losses and accuracy
    values from training. All key variables are initiated at instantiation, and can be retrieved explicitly 
    by referencing them from their parent object (they will be set to None or empty arrays until they have 
    values assigned to them by methods).
    
    '''
    def __init__(self, verbose=False):
        '''
        verbose : bool - 
        '''
        # General
        self.verbose = verbose
        self.__clear = lambda: os.system('cls')

        # Dataset loading
        self.dataset = None
        self.dataloader = None
        self.dataframe = None
        self.stemmer = None
        self.tokenizer = None
        self.max_length = None
        self.validation_dataloader = None
        self.validation_dataset = None

        # vocabulary loading
        self.vocab = None
        self.pad_idx = None
        self.vocab_len = None

        # Model generating
        self.model_name = None
        self.multitask = None
        self.optim = None
        self.scheduler = None
        self.valence_L = None
        self.arousal_L = None
        self.dominance_L = None
        self.quad_L = None
        self.device = None
        self.use_dom = None

        # Training
        self.losses = []
        self.valpoints = []
        self.aropoints = []
        self.true_quads = []
        self.accuracy = []
        self.val_accuracy = [0]

        # Testing
        self.acc_raw = None
        self.acc_am = None
        self.Cmat_raw = None
        self.Cmat_am = None
        self.Cmat_val = None
        self.Cmat_aro = None

    def __printc(self, t):
        self.__clear()
        print(t)
        return

    def load_dataset(self, fname, max_length, batch_size, lyric_col='lyrics', 
                        label_cols = ['valence_tags', 'arousal_tags', 'dominance_tags'], 
                        remove_between_brackets=True, stem=True, stemmer=PorterStemmer(), 
                        tokenize=True, tokenizer=word_tokenize, validation=False):
        '''
        fname : string - filename of excel file to load in
        max_length : int - length to pad/crop sentences to
        batch_size : int - dataloader batch size
        lyric_col : str - name of column with lyrics
        label_cols : [str] - names of columns containing labels
        remove_between_brackets : bool - whether to remove words between brackets
        stem : bool - whether to stem
        stemmer : stemmer - stemmer object
        tokenize : bool - whether to tokenize
        tokenizer : tokenzier - callable tokenizer
        validation : bool - whether the dataset is for validation or not
        '''
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
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        if validation == False:
            self.dataset = dataset
            self.dataloader = dataloader
            self.dataframe = df
            self.max_length = max_length
        else:
            self.validation_dataset = dataset
            self.validation_dataloader = dataloader
            

        if self.verbose:
            print("Succesfully Loaded Dataframe")
        return

    def generate_vocab(self, save=False, save_name=None):
        '''
        save : bool - whether to save the vocab to a pickle
        save_name : str - file name to save vocab to
        '''
        if self.verbose:
            print('Starting Generate Vocab...')
        assert self.dataset is not None, 'Please load in a training dataset before generating a vocabulary'
        assert self.dataframe is not None, 'Please load in a training dataframe before generating a vocabulary'

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
        '''
        fname : str - file name of vocab pkl file to load
        '''
        if self.verbose:
            print('Starting Load Vocab...')
        #assert self.dataset is not None, 'Please load in a dataset before loading a vocabulary'

        self.vocab = Vocabulary()
        self.vocab.load(fname)
        if self.dataset is not None:
            self.dataset.set_vocab(self.vocab)
            print('training vocab loaded')
        if self.validation_dataset is not None:
            self.validation_dataset.set_vocab(self.vocab)
            print('validation vocab loaded')
        self.vocab_len = len(self.vocab)
        self.pad_idx = self.vocab.pad_idx
        if self.verbose:
            print(f'Successfully Loaded {fname} into vocabulary')
        return

    def generate_models(self, emb_size, att_heads, drp, dom, lr, mt_heads, num_enc, 
                        forw_exp, dev, lr_fact=0.2, lr_pat=10, lr_verbose=True, w2v=None, train=True):
        '''
        Generates a new network model with random weights

        Parameters
        ---------------
        emb_size : int - size of embedding vectors to generate
        att_heads : int - number of attention heads (emb_size % att_heads = 0)
        drp : float 0-1 - dropout fraction
        dom : bool - whether to use Dominance dimension in model
        lr : float - learning rate
        mt_heads : int - number of feature layers to generate in final multitask stage
        num_enc : int - number of encoder layers to use
        forw_exp : int - forward expansion rate in feed-forward layer
        dev : Cuda.Device - CPU or GPU
        lr_fact : float - learning rate multiplyer in LR Scheduler
        lr_pat : int - learning rate scheduler patience
        lr_verbose : bool - learning rate scheduler set verbose value
        w2v : torch.Tensor - Tensor containing Word2Vec embedding weights. This will replace the nn.Embedding weights if not set to None
        train : bool - whether this is for training for testing. If testing, no loss functions or optimizers will be generated
        '''
        self.model_name = f'emb{emb_size}att{att_heads}mt{mt_heads}fx{forw_exp}len{self.max_length}drp{str(drp).replace(".","")}dom{1*dom}.pt'
        print(self.model_name)
        if self.verbose:
            print('Starting Generate Models...')
        assert self.vocab_len is not None, 'Please generate or load a vocabulary before training'
        assert self.pad_idx is not None, 'Please generate or load a vocabulary before training'

        self.multitask = mtn.multitaskNet(mt_heads, self.max_length+2, emb_size, drp, dev, 
                                self.vocab_len, num_enc, att_heads, forw_exp, 
                                self.pad_idx, dom, w2v).to(dev)
        self.multitask.double()

        if train:
            self.optim = optim.AdamW(self.multitask.parameters(), lr=lr) # Fine tune this hypPs...
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optim, factor=lr_fact, patience=lr_pat, verbose=lr_verbose)

            self.valence_L = nn.CrossEntropyLoss()
            self.arousal_L = nn.CrossEntropyLoss()
            self.dominance_L = nn.CrossEntropyLoss()
            self.quad_L = nn.CrossEntropyLoss()

        self.device = dev
        self.use_dom = dom
        if self.verbose:
            print('Models Generated Successfully')
        return 

    def load_models(self, fname, lr, lr_fact=0.2, lr_pat=10, lr_verbose=True, train=True):
        '''
        Loads in a pretrained model

        Parameters
        ----------------
        fname : string - filename of model state-dict to be loaded (including .pt)
        lr : learning rate to use
        lr_fact : float - learning rate multiplyer in LR Scheduler
        lr_pat : int - learning rate scheduler patience
        lr_verbose : bool - learning rate scheduler set verbose value
        train : bool - whether this is for training for testing. If testing, no loss functions or optimizers will be generated
        '''
        if self.verbose:
            print('Starting Load Models...')
        self.multitask.load_state_dict(torch.load(fname, map_location=self.device))
        if train:
            self.optim = optim.AdamW(self.multitask.parameters(), lr=lr) # Fine tune this hypPs...
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optim, factor=lr_fact, patience=lr_pat, verbose=lr_verbose)
        if self.verbose:
            print('Models Loaded')
        return

    def VA_to_quadrant(self, V, A):
        '''
        V : float - Valence value
        A : float - Arousal value
        '''
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

    def train(self, epochs, print_step, save_step, save=True, save_name=None, save_epochs=None, 
                show_preds=False, show_acc=True, show_loss=True, show_time=True, 
                enc_version=1, validation_freq=None, val_acc=True, val_cm=True, val_prf=False):
        '''
        Trains the network

        Parameters
        --------------
        epochs : int - number of epochs to train
        print_step : int - print data every N batches (averaged since previous print)
        save_step : int - save data every N batches (averaged since previous save)
        save_name : string - filename to save model under (including .pt)
        save_epochs : int - every Nth epoch the model weights will be saved.
        show_preds : bool - whether to print predictions on print_step
        show_acc : bool - whether to print accuracy on print_step
        show_loss : bool - whether to print loss on print_step
        show_time : bool - whether to print the time taken per epoch
        enc_version : binary - 0=Pytorch implementation of transformer, 1=Manually coded transformer
        validation_freq : int - every Nth epoch the model will run a validation test. Ensure validation data is loaded
        val_acc : bool - whether to print accuracy after validation run
        val_cm : bool - whether to print confusion matrices after validation run
        val_prf : bool - whether to print precision/recall/F-score after validation run
        '''
        if self.verbose:
            print(f'Number of batches per epoch: {len(self.dataloader)}')
            print(f'Printing every {print_step} batches, saving every {save_step} batches')

        # Training loop
        self.multitask.train()
        for epoch in range(epochs):
            total = 0
            correct = 0
            TOTAL = 0
            CORRECT = 0
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
                dominance_loss = self.dominance_L(output[2], dom) if self.use_dom == True else torch.tensor([0]).to(self.device)
                quad_loss = self.quad_L(quad_pred, quad)
                loss = arousal_loss + valence_loss + dominance_loss + quad_loss
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(multitask.parameters(), max_norm=1)
                self.optim.step()
                epoch_l.append(loss.item())
                # Calcuate Accuracy
                total += inp_data.shape[0]
                TOTAL += inp_data.shape[0]
                for i in range(inp_data.shape[0]):
                    correct += 1 if np.argmax(quad_pred[i].detach().cpu().numpy()) == quad[i] else 0
                    CORRECT += 1 if np.argmax(quad_pred[i].detach().cpu().numpy()) == quad[i] else 0

                if (batch_idx + 1) % print_step == 0:
                    if self.verbose:
                        print(f'Batch {batch_idx + 1} / {len(self.dataloader)}')
                    if show_preds:
                        print('VAL pred/true', output[0, :2].detach().cpu().numpy(), val[:2].detach().cpu().numpy())
                        print('ARO pred/true', output[1, :2].detach().cpu().numpy(), aro[:2].detach().cpu().numpy())
                        print('DOM pred/true', output[2, :2].detach().cpu().numpy(), dom[:2].detach().cpu().numpy())
                        print('Quadrant pred/true', np.argmax(quad_pred.detach().cpu().numpy(), axis=1), quad.detach().cpu().numpy())
                    if show_loss:
                        print(f'{print_step} batch average loss:', np.average(epoch_l[-print_step:]))
                    if show_acc:
                        print(f'Batch Accuracy: {100 * correct / total:.2f}%')
                    print('')
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
                print('')
            if show_acc:
                print(f'Epoch Accuracy: {100 * CORRECT / TOTAL:.2f}%')

            if epoch % validation_freq == 0:
                self.optim.zero_grad()
                self.val_accuracy.append(self.test(enc_version, val_acc, val_cm, val_prf, show_progress=False, ret_acc=True))
                self.optim.zero_grad()
            else:
                self.val_accuracy.append(self.val_accuracy[-1])
        
        # Trainig Loop Complete
            if save_epochs is not None:
                if (epoch + 1) % save_epochs == 0:
                    if save_name is not None:
                        epoch_name = save_name[:-3]+ "epoch" + str(epoch + 1) + ".pt"
                    else:
                        epoch_name = self.model_name[:-3]+ "epoch" + str(epoch + 1) + ".pt"
                
                    torch.save(self.multitask.state_dict(), epoch_name)
                    if self.verbose:
                        print(f'Successfully saved model weights for epoch {epoch +1}.')
       
        # Training Loop Complete
        if save:
            torch.save(self.multitask.state_dict(), save_name if save_name is not None else self.model_name)
            if self.verbose:
                print(f'Successfully saved model weights.')
        else:
            print('You have set SAVE to False. Are you sure?')
            name = input('If you wish to save it, please type a file name now (.pt), or enter Y to use a default file name, otherwise enter N: ')
            if name not in ['n', 'N', ' n', ' N', 'no', 'No']:
                if name not in ['Y', 'y']:
                    torch.save(self.multitask.state_dict(), name)
                else:
                    torch.save(self.multitask.state_dict(), self.model_name)
                if self.verbose:
                    print(f'Successfully saved model weights as {name}')
            else:
                print(f'Training Compeleted. Model Not Saved. Total time: {time.time() - t_0:.0f}s')
        return

    def ArgMax_to_quadrant(self, V, A):
        '''
        Takes in the argmaxes for valence and arousal
        1 = positive, 0 = negative  
        '''
        quads = []
        d = {'0,0':2, '0,1':1, '1,1':0, '1,0':3}
        for v, a in zip(V, A):
            b = f'{int(v)},{int(a)}'
            quads.append(d[b])
        return torch.tensor(quads)

    def p_r_f(self, C):
        """
        Calculates precision, recall and f-score from a confusion matrix
        """
        
        if C.shape == (2,2):
            TN,FP,FN,TP = C.ravel()
        else:
            TP = np.diag(C)
            FP = np.sum(C, axis=0) - TP
            FN = np.sum(C, axis=1) - TP
            print(TP, FP, FN)

        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        f_score = 2*precision*recall/(precision+recall)
        
        return precision, recall, f_score

    def test(self, enc_version=1, prnt_acc=True, prnt_cm=True, prnt_prf=True, show_progress=True, ret_acc=False):
        '''
        Performs and evaluation run on the dataset, recording accuracy, confusion matrices and Precision/Recall/F-score.
        enc_version : 0 or 1 - 0 -> pytorch implementation of transformer, 1 -> manually-coded transformer
        prnt_... : Bool - whether to print the specified statistic at the end of testing
        '''
        self.test_losses = []
        # Testing Loop
        self.multitask.eval()
        total = 0
        correct_raw = 0
        correct_am = 0
        correct = 0

        #confusion matrix
        self.Cmat_raw = np.zeros((4,4))
        self.Cmat_am = np.zeros((4,4))

        self.Cmat_val = np.zeros((2,2))
        self.Cmat_aro = np.zeros((2,2))

        labels = [0,1,2,3]

        for batch_idx, batch in enumerate(self.validation_dataloader):
            if show_progress:
                self.__printc('Testing:')
                print(f'{100 * batch_idx / len(self.dataloader):.1f}% Complete')
            
            inp_data = batch['lyrics'].to(self.device)
            val = batch['valence_tags'].long().to(self.device)
            aro = batch['arousal_tags'].long().to(self.device)
            dom = batch['dominance_tags'].long().to(self.device)
            quad = self.VA_to_quadrant(val, aro).to(self.device)

            output, quad_pred_raw = self.multitask(inp_data, version=enc_version)

            val_pred = torch.argmax(output[0], dim=1)
            aro_pred = torch.argmax(output[1], dim=1)
            quad_pred_am = self.ArgMax_to_quadrant(val_pred, aro_pred).numpy()
            quad_pred_raw = torch.argmax(quad_pred_raw, dim=1).detach().cpu().numpy()
            quad = quad.detach().cpu().numpy()

            total += inp_data.shape[0]
            for i in range(inp_data.shape[0]):
                correct += 1 if quad_pred_raw[i] == quad[i] else 0

            correct_raw += sum(quad_pred_raw == quad)
            self.Cmat_raw += confusion_matrix(quad, quad_pred_raw,labels=labels)
            
            correct_am += sum(quad_pred_am == quad)
            self.Cmat_am += confusion_matrix(quad, quad_pred_am,labels=labels)
            #total += len(batch)
            
            self.Cmat_val += confusion_matrix(val.cpu(), val_pred.cpu(),labels=[0,1])
            self.Cmat_aro += confusion_matrix(aro.cpu(), aro_pred.cpu(),labels=[0,1])

        p_raw, r_raw, f_raw = self.p_r_f(self.Cmat_raw)
        p_am, r_am, f_am = self.p_r_f(self.Cmat_am)
        p_val, r_val, f_val = self.p_r_f(self.Cmat_val)
        p_aro, r_aro, f_aro = self.p_r_f(self.Cmat_aro)

        self.acc_raw = 100 * correct_raw / total
        self.acc_am = 100 * correct_am / total

        if prnt_acc:
            print(f'Accuracy: {100 * correct / total:.2f}%')
            print(f'Accuracy of base quadrant predictions: {100 * correct_raw / total:.4f}%')
            print(f'Accuracy of VA quadrant predictions: {100 * correct_am / total:.4f}%')
        if prnt_cm:
            print('Confusion matrix of base quadrant predictions:',self.Cmat_raw)
            print('Confusion matrix of VA quadrant predictions:',self.Cmat_am)
            print('Confusion matrix of valence predictions:',self.Cmat_val)
            print('Confusion matrix of arousal predictions:',self.Cmat_aro)
        if prnt_prf:
            print('Per-label precision, recall, and f-score of base quadrant predictions: {},{},{}'.format(np.round(p_raw,3),np.round(r_raw,3),np.round(f_raw,3)))
            print('Per-label precision, recall, and f-score of VA quadrant predictions: {},{},{}'.format(np.round(p_am,3),np.round(r_am,3),np.round(f_am,3)))
            print('Precision, recall, and f-score valence predictions: {},{},{}'.format(round(p_val,3),round(r_val,3),round(f_val,3)))
            print('Precision, recall, and f-score of arousal predictions: {},{},{}'.format(round(p_aro,3),round(r_aro,3),round(f_aro,3)))
        self.multitask.train()
        if ret_acc:
            return (correct_raw + correct_am) / 2
        return

                
    def return_train_values(self, losses=True, acc=False, val_preds=False, aro_preds=False):
        '''
        returns the values set to true from training.
        '''
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

    def return_test_values(self, acc=True, cm=True, prf=True):
        '''
        returns the values set to true from testing/validation.
        '''
        p_raw, r_raw, f_raw = self.p_r_f(self.Cmat_raw)
        p_am, r_am, f_am = self.p_r_f(self.Cmat_am)
        p_val, r_val, f_val = self.p_r_f(self.Cmat_val)
        p_aro, r_aro, f_aro = self.p_r_f(self.Cmat_aro)
        
        out = ([self.acc_raw, self.acc_raw], [self.Cmat_raw, self.Cmat_am, self.Cmat_val, self.Cmat_aro], 
                [[p_raw, r_raw, f_raw], [p_am, r_am, f_am], [p_val, r_val, f_val], [p_aro, r_aro, f_aro]])

        ind = np.where(np.array([acc, cm, prf]) * 1 == 1)
        return tuple(out[i] for i in ind[0][:])

    def plot_data(self, averaging_window=20, validation=False):
        '''
        Plots losses and accuracy.
        averaging_window : int - size of moving average frame
        '''
        w = np.ones(averaging_window) / averaging_window
        fig, axs = plt.subplots(2)
        axs[0].plot(np.convolve(self.losses, w)[averaging_window:-averaging_window])
        axs[0].set_title('Training Losses')
        axs[1].plot(np.convolve(self.accuracy, w)[averaging_window:-averaging_window])
        axs[1].set_title('Quadrant Prediction Accuracy')
        if validation:
            axs[1].plot(self.val_accuracy[averaging_window:-averaging_window])
            
        plt.show()
        return

def get_quad(df,A_thresh=5,V_thresh=5):
    """Returns quadrant given VAD scores"""
    valence = df['valence_tags']
    arousal = df['arousal_tags']
    if arousal>A_thresh and valence>V_thresh:
        return 'UR'
    elif arousal>A_thresh and valence<=V_thresh:
        return 'UL'
    elif arousal<=A_thresh and valence<=V_thresh:
        return 'LL'
    elif arousal<=A_thresh and valence>V_thresh:
        return 'LR'


def generate_test_val(dataframe, split, fnames, type='excel',oversample=False):
    '''
    Creates seperate files for train and val sets.

    Parameters
    ---------------------
    dataframe : Pandas.DataFrame - dataframe to be split
    split : float 0 < s < 1 - proportion to be assigned to train
    fnames : list<string> - name for each train and val file, inluding extension
    type : string - type of save format (excel, csv, pickle)
    oversample: Bool - whether to oversample for class imbalance
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

    if oversample==True:
        #determine quadrant
        y=train.apply(get_quad,axis=1)
        #intialise oversampler
        ros = RandomOverSampler()
        #update training data with oversampling
        train, Y_bal = ros.fit_resample(train,y)


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



