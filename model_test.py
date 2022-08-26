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
import random
import math

#define variables
seed = input("Starting Phrase (Five Words): ")
quantity = int(input("Quantity: "))
sequence_length = 6
#weirdness = int(input("Obscurity [0.1-2.0]: "))
weirdness = 1.9
model = keras.models.load_model('trained_25_epochs')

sentence = seed.split(" ")

#fetch dictionaries of learned vocabulary
with open('word_indices.pkl', 'rb') as f:
    word_indices = pickle.load(f)

with open('indices_word.pkl', 'rb') as f:
    indices_word = pickle.load(f)

# Function from keras-team/keras/blob/master/examples/lstm_text_generation.py
# Diversification function to encourage choice of more obscure vocabulary, 
# avoid overusing common words
def sample(preds, temperature=weirdness):
    preds = np.asarray(preds).astype('float64')

    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


#Loop to generate quantity number of words
for i in range(quantity):

    # generate line breaks, the network can read them but we do not let it
    # write them because it liked them too much
    if random.random() < 0.1:
        sentence = sentence[1:]
        sentence.append("\n")
        print(" "+"\n", end="")
        continue

    # read the last 5 words and predict the next based on them
    x_pred = np.zeros((1, sequence_length), dtype=np.int32)
    for t, word in enumerate(sentence):
        x_pred[0, t] = word_indices[word]

    preds = model.predict(x_pred, verbose=0)[0]

    next_index = sample(preds[1:]) + 1

    next_word = indices_word[next_index]

    sentence = sentence[1:]
    
    sentence.append(next_word)
    print(" "+next_word, end="")

    

print("\n")

