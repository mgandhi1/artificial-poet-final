import numpy as np
import pickle
import pandas as pd
import seaborn as sns
import sys
import os 
import io
import codecs
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional, Embedding, Activation
from keras.callbacks import ModelCheckpoint, EarlyStopping, LambdaCallback

import math

sns.set()

data_raw = pd.read_csv(r"PoetryFoundationData.csv")

corpus = "allPoems.txt"
with io.open(corpus, encoding='utf-8') as f:
  text = f.read().lower().replace('\n', ' \n ')
print('Corpus length in characters:', len(text))

text_in_words = [w for w in text.split(' ') if w.strip() != '' or w == '\n']
print('Corpus length in words:', len(text_in_words))

MIN_WORD_FREQUENCY=10

# Calculate word frequency
word_freq = {}
for word in text_in_words:
    word_freq[word] = word_freq.get(word, 0) + 1

ignored_words = set()
for k, v in word_freq.items():
    if word_freq[k] < MIN_WORD_FREQUENCY:
        ignored_words.add(k)

words = set(text_in_words)
print('Unique words before ignoring:', len(words))
print('Ignoring words with frequency <', MIN_WORD_FREQUENCY)
words = sorted(set(words) - ignored_words)
print('Unique words after ignoring:', len(words))

word_indices = dict((c, i) for i, c in enumerate(words))
indices_word = dict((i, c) for i, c in enumerate(words))

with open('word_indices.pkl', 'wb') as f:
    pickle.dump(word_indices, f)

with open('indices_word.pkl', 'wb') as f:
    pickle.dump(indices_word, f)