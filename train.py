def sampled_loss(y_true, y_pred):
    labels = tf.reshape(y_true, [-1, 1])
    # We need to compute the sampled_softmax_loss using 32bit floats to
    # avoid numerical instabilities.
    local_w_t = tf.cast(w_t, tf.float32)
    local_b = tf.cast(b, tf.float32)
    local_inputs = tf.cast(y_pred, tf.float32)
    return tf.cast(
        tf.nn.sampled_softmax_loss(local_w_t, local_b, local_inputs, labels,
                                   25, vocab_size),
        np.float32)

import numpy as np

import tensorflow as tf
from data_utils import Vocabulary, Dataset

from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, TimeDistributedDense, Activation
from keras.utils.visualize_util import plot

vocab = Vocabulary.from_file("1b_word_vocab.txt")
print("loaded vocab, num tokens " + str(vocab._num_tokens))

print("Loading data...")
dataset = Dataset(vocab, "/mnt/raid1/billion-word-corpus/1-billion-word-language-modeling-benchmark/training-monolingual.tokenized.shuffled/*")
print("Data loaded")

seq_len = 25  # cut texts after this number of words (among top max_features most common words)
batch_size = 32
embed_size = 128
vocab_size = vocab._num_tokens

it = dataset.iterate_once(batch_size,seq_len)

input = Input(shape=(seq_len,), dtype='int32', name='main_input')
f = Embedding(vocab_size+1, embed_size, input_length=seq_len)(input)
f = LSTM(output_dim=256,return_sequences=True)(f)
f = TimeDistributedDense(vocab_size, activation='relu')(f)
f = Activation(activation='softmax')(f)
model = Model(input,f)
model.compile(loss='categorical_crossentropy', optimizer='adam')
plot(model, to_file='model.png')

for i, (x, y, w) in enumerate(it):
    print(str(i))
    print("logloss = " + str(model.train_on_batch(x, y)))