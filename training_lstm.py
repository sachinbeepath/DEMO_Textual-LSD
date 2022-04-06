import torch
import utils
import torch.nn.functional as F

##### Key Variables #####
# Hashed values are those used in the reference paper
EPOCHS = 40
BATCH_SIZE = 128
LR = 3e-4 #2e-5
# USE_VAL = False
# USE_AROU = False
USE_DOM = True

# USE_QUAD = True
FILENAME = 'Datasets/train_balanced.xlsx'
VAL_FILENAME = 'Datasets/validation_bal_3Apr.xlsx'
ATTENTION_HEADS = 8
EMBEDDING_SIZE = 32
NUM_ENCODER_LAYERS = 1
FORWARD_XP = 32
DROPOUT = 0.4
MAXLENGTH = 128
MT_HEADS = 8
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using ', DEVICE)

PRINT_STEP = 25
SAVE_STEP = 10
w2v = torch.load('w2v_window10_min10_iter10.pkl')
w2v.requires_grad = False

#%%

trainer = utils.Textual_LSD_TVT(verbose=True)
trainer.load_dataset(FILENAME, MAXLENGTH, BATCH_SIZE)
trainer.load_dataset(VAL_FILENAME, MAXLENGTH, BATCH_SIZE, validation=True)

trainer.load_vocab('Pickles/NewVocab.pkl')

#%%

# trainer.generate_models(EMBEDDING_SIZE, ATTENTION_HEADS, DROPOUT, USE_VAL, USE_AROU, USE_DOM, USE_QUAD,
#                         LR, MT_HEADS, NUM_ENCODER_LAYERS, FORWARD_XP, DEVICE, lr_pat=15, w2v=w2v, version=2)

trainer.generate_models_lstm(EMBEDDING_SIZE,BATCH_SIZE, DROPOUT,USE_DOM, LR, DEVICE, lr_pat=15)#, w2v=w2v)

trainer.train(EPOCHS, PRINT_STEP, SAVE_STEP, enc_version=1,save_epochs=2,validation_freq=1, val_prf=True)
trainer.plot_data(averaging_window=1)

trainer.plot_data(averaging_window=1,validation=True)