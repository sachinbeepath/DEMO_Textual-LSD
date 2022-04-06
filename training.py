import torch
import utils
import torch.nn.functional as F

import torch.nn.functional as F

##### Key Variables #####
# Hashed values are those used in the reference paper
<<<<<<< HEAD
EPOCHS = 4 #Until convergence
BATCH_SIZE = 128 # 8
=======
EPOCHS = 30 #Until convergence
BATCH_SIZE = 128
>>>>>>> 21878442c01f7f7c8ba02f152d5899c817cd9d59
LR = 3e-4 #2e-5
USE_DOM = False
FILENAME = 'Datasets/train_balanced.xlsx'
<<<<<<< HEAD
VFILENAME = 'Datasets/validation_bal_3Apr.xlsx'
ATTENTION_HEADS = 8 # 8
EMBEDDING_SIZE = 32 # 512
NUM_ENCODER_LAYERS = 1 # 3
FORWARD_XP = 3
DROPOUT = 0.35 # 0.1
MAXLENGTH = 128 #1024
MT_HEADS = 64 # 8
=======
VAL_FILEANAME= "Datasets/validation_bal_3Apr.xlsx"
ATTENTION_HEADS = 8
EMBEDDING_SIZE = 32
NUM_ENCODER_LAYERS = 1
FORWARD_XP = 16
DROPOUT = 0.45
MAXLENGTH = 128
MT_HEADS = 8
>>>>>>> 21878442c01f7f7c8ba02f152d5899c817cd9d59
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using ', DEVICE)

PRINT_STEP = 25
SAVE_STEP = 10
w2v = torch.load('w2v_window10_min0_iter20.pkl')
w2v.requires_grad= False
print(w2v.shape)

trainer = utils.Textual_LSD_TVT(verbose=True)
trainer.load_dataset(FILENAME, MAXLENGTH, BATCH_SIZE)
trainer.load_dataset(VFILENAME, MAXLENGTH, BATCH_SIZE, validation=True)
#trainer.load_vocab('vocab_emb64.pkl')
trainer.load_vocab('Pickles/NewVocab.pkl')
trainer.generate_models(EMBEDDING_SIZE, ATTENTION_HEADS, DROPOUT, USE_DOM,
                        LR, MT_HEADS, NUM_ENCODER_LAYERS, FORWARD_XP, DEVICE, lr_pat=15)
trainer.train(EPOCHS, PRINT_STEP, SAVE_STEP, enc_version=2, save_epochs=5, validation_freq=1)
trainer.plot_data(averaging_window=5)



