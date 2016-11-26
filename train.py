from data_utils import Dataset
from keras.utils import generic_utils
import numpy as np


data_dir = '/mnt/raid1/billion-word-corpus/1-billion-word-language-modeling-benchmark/training-monolingual.tokenized.shuffled/'
NUM_WORDS = 20000

seq_len = 25
dataset = Dataset(data_dir,NUM_WORDS)
X,Y = dataset.create_X_Y(seq_len=seq_len,one_hot_y=False)

params = {}
params['vocab_size'] = dataset.vocab_size
params['num_classes'] = dataset.vocab_size

batch_size = 32
embed_size = 128
num_epochs = 2


losses,outputs = get_model_tf(params)
sess.run(tf.initialize_all_variables())
loss_function = tf.reduce_mean(losses)
train_step = tf.train.AdamOptimizer(0.001).minimize(loss_function)

train_step.run(feed_dict={input: X_batch, labels: Y[start:end]})


progbar = generic_utils.Progbar(len(X))
for epoch in range(num_epochs):
    batches = range(0,len(X)/batch_size)
    for batch in batches[:-1]:
        start = batch*batch_size
        end = (batch+1)*batch_size
        X_batch = X[start:end]
        Y_batch = dataset.Y_to_Categorical(Y[start:end])
        loss = model.train_on_batch(X_batch,Y_batch)
        perp = np.exp(loss)
        progbar.add(len(X_batch), values=[("train loss", loss),("train perplexity", perp)])
