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
from data_utils import Dataset

import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, TimeDistributedDense, Activation

data_dir = '/mnt/raid1/billion-word-corpus/1-billion-word-language-modeling-benchmark/training-monolingual.tokenized.shuffled/'
NUM_WORDS = 20000

seq_len = 25
dataset = Dataset(data_dir,NUM_WORDS)
X,Y = dataset.create_X_Y(seq_len=seq_len,one_hot_y=True)

batch_size = 32
embed_size = 128

it = dataset.iterate_once(batch_size,seq_len)

input = Input(shape=(seq_len,), dtype='int32', name='main_input')
f = Embedding(vocab_size+1, embed_size, input_length=seq_len)(input)
f = LSTM(output_dim=256,return_sequences=True)(f)
f = TimeDistributedDense(vocab_size, activation='relu')(f)
f = Activation(activation='softmax')(f)
model = Model(input,f)
model.compile(loss='categorical_crossentropy', optimizer='adam')
plot(model, to_file='model.png')

