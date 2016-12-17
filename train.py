from data_utils import Dataset
from keras.utils import generic_utils
from model import LanguageModel
import numpy as np
import tensorflow as tf
import time as time

tf.logging.set_verbosity(tf.logging.ERROR)

data_dir = '/mnt/raid1/billion-word-corpus/1-billion-word-language-modeling-benchmark/training-monolingual.tokenized.shuffled/'
valid_data_dir = '/mnt/raid1/billion-word-corpus/1-billion-word-language-modeling-benchmark/heldout-monolingual.tokenized.shuffled/'
save_dir = '/home/ab455/language-model/checkpoints/'
num_words = None

seq_len = 25
batch_size = 512
valid_batch_size = 16 ## Needs to be smaller due to memory issues
embed_size = 256
num_epochs = 5

dataset = Dataset(data_dir,num_words)
dataset.set_batch_size(batch_size)
dataset.set_seq_len(seq_len)

params = {}
params['vocab_size'] = dataset.vocab_size
params['num_classes'] = dataset.vocab_size
params['batch_size'] = batch_size
params['seq_len'] = seq_len
params['hidden_dim'] = 512
params['num_layers'] = 1

model = LanguageModel(params)
model.compile()
n_valid_batches = 20
for epoch in range(num_epochs):
    dataset.set_data_dir(data_dir)
    dataset.set_batch_size(batch_size)
    progbar = generic_utils.Progbar(dataset.token.document_count)
    for X_batch,Y_batch in dataset:
        t0 = time.time()
        loss = model.train_on_batch(X_batch,Y_batch)
        perp = np.exp(np.float32(loss))
        t1 = time.time()
        wps = np.round((batch_size * seq_len)/(t1-t0))
        progbar.add(len(X_batch), values=[("loss", loss),("perplexity", perp),("words/sec", wps)])
    model.save(save_dir)
    dataset.set_data_dir(valid_data_dir)
    dataset.set_batch_size(valid_batch_size)
    valid_perp = []
    count = 0
    if n_valid_batches is not None:
        progbar = generic_utils.Progbar(n_valid_batches)
    print '\n\nEstimating validation perplexity on ' + str(n_valid_batches) + ' batches (' + str(n_valid_batches*batch_size) + ' samples)'
    for X_batch, Y_batch in dataset:
        if n_valid_batches is not None:
            progbar.add(1)
        valid_perp.append(model.evaluate(X_batch, Y_batch))
        count += 1
        if count > n_valid_batches:
            break
    print '\nEstimated Validation Perplexity: ' + str(np.mean(valid_perp)) + '\n'
