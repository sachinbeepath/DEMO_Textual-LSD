#%%
from cProfile import label
from operator import concat
from re import A
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import words as ws
import matplotlib.pyplot as plt
import matplotlib.colors
from collections import Counter
#Spacy?

%matplotlib inline

datafile = pd.read_pickle("Data_8500_cleaned.pkl")
df = pd.DataFrame(datafile)

#%%

scaled_df = pd.DataFrame()
for tag in ['valence_tags','arousal_tags','dominance_tags']:
    scaled_df[tag] = (df[tag]-5)/5
    min_ = (np.min(df[tag]) - 5)/5
    max_ = (np.max(df[tag]) - 5)/5


    print(f'min of {tag} is {min_}, max is {max_}')
#%%

median_scaled_df = pd.DataFrame()
for tag in ['valence_tags','arousal_tags','dominance_tags']:
    min_ = np.min(df[tag])
    max_ = np.max(df[tag])
    range_ = max_-min_
    median_scaled_df[tag] = (df[tag]-np.median(df[tag]))/(0.5*range_)



    print(f'min of {tag} is {min_}, max is {max_}')


#%%
cm = plt.cm.get_cmap('RdYlBu')

scat_plt = plt.scatter(scaled_df['valence_tags'],scaled_df['arousal_tags'],c = scaled_df['dominance_tags'],vmin=-1, vmax=+1,cmap=cm)

cbar = plt.colorbar(scat_plt,)
cbar.set_label('dominance')
plt.xlabel('valence')
plt.ylabel('arousal')
plt.show()


#create functions to get key metrics
#%%
def get_quad(valence,arousal):
    if arousal>0 and valence>0:
        return 'UR'
    elif arousal>0 and valence<=0:
        return 'UL'
    elif arousal<=0 and valence<=0:
        return 'LL'
    elif arousal<=0 and valence>0:
        return 'LR'


def drop_pads(df):
    indicies = (df != '<PAD>')
    return df[indicies]

def count_distinct_words(song):
    return len(set(song))

def get_complexity(song):
    return len(set(song))/len(song)

removed_pads = df['Clean_Lyrics'].apply(drop_pads)
song_lengths = removed_pads.apply(len)
distinct_words = removed_pads.apply(count_distinct_words)
complexity = removed_pads.apply(get_complexity)

print(f'mean song length is {np.mean(song_lengths)}')
print(f'song length std is {np.std(song_lengths)}')

#Get quadrants
#%%
median_scaled_df['quadrant'] = median_scaled_df.apply(lambda row: get_quad(row['valence_tags'],row['arousal_tags']),axis =1)
med_distribution = median_scaled_df.groupby('quadrant').count()
#all tag counts are the same
med_distribution['quadrant'] = med_distribution['arousal_tags']
med_distribution['percentage']= med_distribution['quadrant']/np.sum(med_distribution['quadrant'])
print(med_distribution[['quadrant','percentage']])
#%%
print(med_distribution[['quadrant','percentage']])

#%%
scaled_df['quadrant'] = scaled_df.apply(lambda row: get_quad(row['valence_tags'],row['arousal_tags']),axis =1)
print(scaled_df.head())

distribution = scaled_df.groupby('quadrant').count()
distribution['percentage']= distribution['quadrants']/np.sum(distribution['quadrants'])

print(distribution[['quadrant','percentage']])


#%%
print(10%10)

#plot and save histograms
# %%

plt.hist(song_lengths, bins=20, range = (np.percentile(song_lengths,0),np.percentile(song_lengths,95)))
plt.xlabel('no. of words')
plt.ylabel('no. of songs')
plt.savefig('Figures/song_length.png')
plt.show()

#%%
plt.hist(distinct_words, bins=20, range = (np.percentile(distinct_words,0),np.percentile(distinct_words,95)))
plt.title('distinct_words_per_song')
plt.xlabel('no. of distinct words')
plt.ylabel('no. of songs')
plt.savefig('Figures/distinct_words.png')
plt.show()

#%%
print(get_complexity(removed_pads[0]))

#%%
plt.hist(complexity, bins=20)
plt.title('complexity')
plt.xlabel('distinct words/ total words')
plt.ylabel('no. of songs')
plt.savefig('Figures/complexity.png')
plt.show()

#%%

#group all lyrics together
all_lyrics = []
for i, lyrics in enumerate(removed_pads):
    all_lyrics += lyrics.tolist()

print(f'total words is{len(all_lyrics)}')
print(f'total unique words is {(len(set(all_lyrics)))}')

# %%
word_dict = Counter(all_lyrics)
word_items = word_dict.items()
word_list = list(word_items)
vocab_df_sorted = pd.DataFrame(word_list, columns=['word','count']).sort_values(by =['count'],ascending=False)
vocab_df = pd.DataFrame(np.array(vocab_df_sorted[['word','count']]),columns=['word','count'])
print(vocab_df)