import torch
import utils

##### Key Variables #####
BATCH_SIZE = 32
USE_DOM = True
FILENAME = '8500_songs_validation.xlsx' 
ATTENTION_HEADS = 8
EMBEDDING_SIZE = 64
NUM_ENCODER_LAYERS = 1
FORWARD_XP = 64
DROPOUT = 0.25
MAXLENGTH = 256
MT_HEADS = 8
LABEL_DICT = {'relaxed': 3, 'angry': 1, 'happy': 0, 'sad': 2}
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

PRINT_STEP = 10
SAVE_STEP = 5   

trainer = utils.Textual_LSD_TVT(verbose=True)
trainer.load_dataset(FILENAME, MAXLENGTH, BATCH_SIZE)
trainer.load_vocab('vocab_emb64.pkl')
trainer.generate_models(EMBEDDING_SIZE, ATTENTION_HEADS, DROPOUT, USE_DOM, 
                        None, MT_HEADS, NUM_ENCODER_LAYERS, FORWARD_XP, DEVICE, lr_pat=15, train=False)
trainer.load_models('Previous_40epoch_settings_25drp.pt', None, train=False)
trainer.test()





'''
TRAIN_VAL_SPLIT = 0.8

##### Load Data into Dataset#####
printc(f'Reading in {FILENAME} and creating dataloaders...')
file = pd.read_excel(FILENAME)
dataframe = pd.DataFrame(file)
ps = PorterStemmer()
stemmer = lambda x: ps.stem(x)
dataset = dl.LSD_DataLoader(dataframe, 'lyrics', ['valence_tags', 'arousal_tags', 'dominance_tags'], MAXLENGTH)
dataset.scale_VAD_scores(5, 5)
dataset.clean_lyrics(remove_between_brackets=True, stem=True, stemmer=stemmer, tokenize=True, tokenizer=word_tokenize, length=MAXLENGTH)

##### Vocab work #####
printc('Creating vocabulary from training data...')
english = utils.Vocabulary()
english.load('vocab_emb64_v2.pkl')
PAD_IDX = english.pad_idx
VOCAB_LEN = len(english)
dataset.set_vocab(english)

##### Create Dataloaders #####
dataloader_val = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

##### Prepare Model, Optimizer and Criterion #####
print('Creating Models') #Such a janky solution for weird difference in vocab length. Deffo gonna break
multitask = mtn.multitaskNet(MT_HEADS, MAXLENGTH+2, EMBEDDING_SIZE, DROPOUT, DEVICE, 
                            VOCAB_LEN, NUM_ENCODER_LAYERS, ATTENTION_HEADS, FORWARD_XP, 
                            PAD_IDX, USE_DOM).to(DEVICE)
multitask.double()

# change file name
multitask.load_state_dict(torch.load('MTL_clasification_50eps.pt'))

losses = []
# Testing Loop
multitask.eval()
total = 0
correct_raw = 0
correct_am = 0

#confusion matrix
Cmat_raw = np.zeros((4,4))
Cmat_am = np.zeros((4,4))

Cmat_val = np.zeros((2,2))
Cmat_aro = np.zeros((2,2))


labels = [0,1,2,3]

for batch_idx, batch in enumerate(dataloader_val):
    inp_data = batch['lyrics'].to(DEVICE)
    val = batch['valence_tags'].long().to(DEVICE)
    aro = batch['arousal_tags'].long().to(DEVICE)
    dom = batch['dominance_tags'].long().to(DEVICE)
    quad = VA_to_quadrant(val, aro).to(DEVICE)
    output, quad_pred_raw = multitask(inp_data, version=0)
    val_pred = torch.argmax(output[0], dim=1)
    aro_pred = torch.argmax(output[1], dim=1)
    quad_pred_am = ArgMax_to_quadrant(val_pred, aro_pred).numpy()
    quad_pred_raw = torch.argmax(quad_pred_raw, dim=1).detach().cpu().numpy()
    quad = quad.detach().cpu().numpy()
    
    if (batch_idx + 1) % PRINT_STEP == 0:
        print('')
        print(f'Batch {batch_idx + 1} / {len(dataloader_val)}')
        #print(quad_pred_am, quad_pred_raw, quad)

    correct_raw += sum(quad_pred_raw == quad)
    Cmat_raw += confusion_matrix(quad_pred_raw,quad,labels=labels)
    
    correct_am += sum(quad_pred_am == quad)
    Cmat_am += confusion_matrix(quad_pred_am,quad,labels=labels)
    total += inp_data.shape[0]
    
    Cmat_val += confusion_matrix(val_pred.cpu(),val.cpu(),labels=[0,1])
    Cmat_aro += confusion_matrix(aro_pred.cpu(),aro.cpu(),labels=[0,1])


p_raw, r_raw, f_raw = p_r_f(Cmat_raw)
p_am, r_am, f_am = p_r_f(Cmat_am)
p_val, r_val, f_val = p_r_f(Cmat_val)
p_aro, r_aro, f_aro = p_r_f(Cmat_aro)


print(f'Accuracy of base quadrant predictions: {100 * correct_raw / total:.4f}%')
print(f'Accuracy of VA quadrant predictions: {100 * correct_am / total:.4f}%')

print('Confusion matrix of base quadrant predictions:',Cmat_raw)
print('Confusion matrix of VA quadrant predictions:',Cmat_am)

print('Confusion matrix of valence predictions:',Cmat_val)
print('Confusion matrix of arousal predictions:',Cmat_aro)

print('Per-label precision, recall, and f-score of base quadrant predictions: {},{},{}'.format(np.round(p_raw,3),np.round(r_raw,3),np.round(f_raw,3)))
print('Per-label precision, recall, and f-score of VA quadrant predictions: {},{},{}'.format(np.round(p_am,3),np.round(r_am,3),np.round(f_am,3)))


print('Precision, recall, and f-score valence predictions: {},{},{}'.format(round(p_val,3),round(r_val,3),round(f_val,3)))
print('Precision, recall, and f-score of arousal predictions: {},{},{}'.format(round(p_aro,3),round(r_aro,3),round(f_aro,3)))

'''
