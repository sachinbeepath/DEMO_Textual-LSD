import torch
import utils
import torch.nn.functional as F

##### Key Variables #####
# Hashed values are those used in the reference paper
EPOCHS = 4 #Until convergence
BATCH_SIZE = 128 # 8
LR = 3e-4 #2e-5
USE_DOM = True
FILENAME = 'Datasets/train_balanced.xlsx'
VFILENAME = 'Datasets/validation_bal_3Apr.xlsx'
ATTENTION_HEADS = 8 # 8
EMBEDDING_SIZE = 32 # 512
NUM_ENCODER_LAYERS = 1 # 3
FORWARD_XP = 3
DROPOUT = 0.35 # 0.1
MAXLENGTH = 128 #1024
MT_HEADS = 64 # 8
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using ', DEVICE)

PRINT_STEP = 25
SAVE_STEP = 10
w2v = torch.load('Pickles/w2v_weights.pkl')
w2v = F.normalize(w2v, dim=1)
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



