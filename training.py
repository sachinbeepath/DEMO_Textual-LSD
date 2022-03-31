import torch
import utils
import torch.nn.functional as F

##### Key Variables #####
# Hashed values are those used in the reference paper
EPOCHS = 2 #Until convergence
BATCH_SIZE = 32 # 8
LR = 5e-5 #2e-5
USE_DOM = False
FILENAME = '8500_songs_training.xlsx'
ATTENTION_HEADS = 8 # 8
EMBEDDING_SIZE = 64 # 512
NUM_ENCODER_LAYERS = 1 # 3
FORWARD_XP = 64
DROPOUT = 0.25 # 0.1
MAXLENGTH = 256 #1024
MT_HEADS = 8 # 8
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using ', DEVICE)

PRINT_STEP = 25
SAVE_STEP = 10
w2v = torch.load('w2v_weights.pkl')
w2v = F.normalize(w2v, dim=1)
print(w2v.shape)

trainer = utils.Textual_LSD_TVT(verbose=True)
trainer.load_dataset(FILENAME, MAXLENGTH, BATCH_SIZE)
trainer.load_vocab('vocab_emb64.pkl')
<<<<<<< HEAD
trainer.generate_models(EMBEDDING_SIZE, ATTENTION_HEADS, DROPOUT, USE_DOM, 
                        LR, MT_HEADS, NUM_ENCODER_LAYERS, FORWARD_XP, DEVICE, lr_pat=25)
trainer.load_models('Previous_40epoch_settings_25drp.pt', None, train=False)
trainer.train(EPOCHS, PRINT_STEP, SAVE_STEP, 'nodomtest.pt', enc_version=1)
=======
trainer.generate_models(EMBEDDING_SIZE, ATTENTION_HEADS, DROPOUT, USE_DOM,
                        LR, MT_HEADS, NUM_ENCODER_LAYERS, FORWARD_XP, DEVICE, lr_pat=15)
trainer.train(EPOCHS, PRINT_STEP, SAVE_STEP, 'Test.pt', enc_version=1)
>>>>>>> 9a4966a128d2cafaac6b94e604dc103e7a326b23
trainer.plot_data(averaging_window=1)





<<<<<<< HEAD



=======
>>>>>>> 9a4966a128d2cafaac6b94e604dc103e7a326b23
