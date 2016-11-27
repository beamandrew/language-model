from data_utils import Dataset
from keras.utils import generic_utils
from model import LanguageModel
import numpy as np

data_dir = '/mnt/raid1/billion-word-corpus/1-billion-word-language-modeling-benchmark/training-monolingual.tokenized.shuffled/'
NUM_WORDS = 125000

seq_len = 25
dataset = Dataset(data_dir,NUM_WORDS)
X,Y = dataset.create_X_Y(seq_len=seq_len,one_hot_y=False)

batch_size = 32
embed_size = 256
num_epochs = 2

params = {}
params['vocab_size'] = dataset.vocab_size
params['num_classes'] = dataset.vocab_size
params['batch_size'] = batch_size
params['seq_len'] = seq_len
params['hidden_dim'] = 128

model = LanguageModel(params)
model.compile()

progbar = generic_utils.Progbar(len(X))
for epoch in range(num_epochs):
    batches = range(0,len(X)/batch_size)
    for batch in batches[:-1]:
        start = batch*batch_size
        end = (batch+1)*batch_size
        X_batch = X[start:end]
        Y_batch = Y[start:end]
        loss = model.train_on_batch(X_batch,Y_batch)
        perp = np.exp(np.float32(loss))
        progbar.add(len(X_batch), values=[("train loss", loss),("train perplexity", perp)])
