from data_utils import Dataset
from keras.utils import generic_utils
from model import LanguageModel
import numpy as np
import tensorflow as tf
import time as time

tf.logging.set_verbosity(tf.logging.ERROR)

data_dir = '/mnt/raid1/billion-word-corpus/1-billion-word-language-modeling-benchmark/training-monolingual.tokenized.shuffled/'
NUM_WORDS = 125000

seq_len = 25
dataset = Dataset(data_dir,NUM_WORDS)
X,Y = dataset.create_X_Y(seq_len=seq_len,one_hot_y=False)

batch_size = 256
embed_size = 256
num_epochs = 10

params = {}
params['vocab_size'] = dataset.vocab_size
params['num_classes'] = dataset.vocab_size
params['batch_size'] = batch_size
params['seq_len'] = seq_len
params['hidden_dim'] = 512
params['num_layers'] = 2

model = LanguageModel(params)
model.compile()

progbar = generic_utils.Progbar(len(X))
for epoch in range(num_epochs):
    batches = range(0,len(X)/batch_size)
    for batch in batches[:-1]:
        t0 = time.time()
        start = batch*batch_size
        end = (batch+1)*batch_size
        X_batch = X[start:end]
        Y_batch = Y[start:end]
        #perp =  model.evaluate(X_batch,Y_batch)
        loss = model.train_on_batch(X_batch,Y_batch)
        perp = np.exp(np.float32(loss))
        t1 = time.time()
        wps = np.round((batch_size * seq_len)/(t1-t0))
        progbar.add(len(X_batch), values=[("loss", loss),("perplexity", perp),("words/sec", wps)])


