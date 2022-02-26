import pandas as pd
import numpy as np
import nltk
from nltk.corpus import words as ws
#Spacy?

datafile = pd.read_excel("Data_8500_songs.xlsx")
df = pd.DataFrame(datafile)

df = df[['Artist', 'song', 'valence_tags', 'arousal_tags', 'dominance_tags', 'lyrics']]
print(df.head())




