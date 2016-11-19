import numpy as np

import tensorflow as tf
from data_utils import Vocabulary, Dataset

from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, Bidirectional


vocab = Vocabulary.from_file("1b_word_vocab.txt")
print("loaded vocab, num tokens " + str(vocab._num_tokens))

print("Loading data...")
dataset = Dataset(vocab, "/mnt/raid1/billion-word-corpus/1-billion-word-language-modeling-benchmark/training-monolingual.tokenized.shuffled/*")
print("Data loaded")

seq_len = 25  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

it = dataset.iterate_once(batch_size,seq_len)

main_input = Input(shape=(seq_len,), dtype='int32', name='main_input')
model.add(Embedding(max_features, 128, input_length=maxlen))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))


for i, (x, y, w) in enumerate(it):
    print('x' + str(x.shape))
    print('y' + str(y.shape))
    print('w' + str(w.shape))