import torch

##### Key Variables #####
# Hashed values are those used in the reference paper
EPOCHS = 50 #Until convergence
BATCH_SIZE = 64 # 8
LR = 3e-3 #2e-5
USE_DOM = True
FILENAME = '8500_songs_training.xlsx'
ATTENTION_HEADS = 8 # 8
EMBEDDING_SIZE = 32 # 512
NUM_ENCODER_LAYERS = 1 # 3
FORWARD_XP = 16
DROPOUT = 0.25 # 0.1
MAXLENGTH = 128 #1024
MT_HEADS = 8 # 8
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using ', DEVICE)

PRINT_STEP = 50
SAVE_STEP = 5 







