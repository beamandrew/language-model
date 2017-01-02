import os
import string
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.utils import generic_utils

import pickle

class DataGenerator(object):
    def __init__(self,data_dir):
        self.file_list = os.listdir(data_dir)
        self.data_dir = data_dir
        self.progbar = generic_utils.Progbar(len(self.file_list))
    def __iter__(self):
        for fname in self.file_list:
            self.progbar.add(1)
            with open(os.path.join(self.data_dir,fname)) as f:
                for line in f:
                    yield line


class Dataset(object):
    def __init__(self,data_dir,num_words=None,batch_size=None,seq_len=None):
        self.file_list = os.listdir(data_dir)
        self.data_dir = data_dir
        self.texts = DataGenerator(data_dir)
        self.token = Tokenizer(nb_words=num_words, lower=True, split=' ')
        print('Reading files...')
        self.token.fit_on_texts(self.texts)
        self.vocab_size = len(self.token.word_index)
        if num_words is not None:
            self.vocab_size = num_words
        self.batch_size = batch_size
        self.seq_len = seq_len
    def  __iter__(self):
        for fname in self.file_list:
            with open(os.path.join(self.data_dir, fname)) as f:
                lines = []
                for line in f:
                    if len(lines) == self.batch_size:
                        proc_txt = self.token.texts_to_sequences(lines)
                        X_batch = [txt[:min(len(txt)-1, self.seq_len)] for txt in proc_txt]
                        X_batch = pad_sequences(X_batch,maxlen=self.seq_len)
                        Y_batch = [txt[1:min(len(txt), self.seq_len) + 1] for txt in proc_txt]
                        Y_batch = pad_sequences(Y_batch,maxlen=self.seq_len)
                        lines = []
                        yield X_batch,Y_batch
                    else:
                        lines.append(line)
    def set_batch_size(self,batch_size):
        self.batch_size = batch_size
    def set_seq_len(self, seq_len):
        self.seq_len = seq_len
    def set_data_dir(self,data_dir):
        self.data_dir = data_dir
        self.file_list = os.listdir(data_dir)
    def save(self,save_path='./'):
        ## Need to save the tokenizer properties ##
        with open(save_path + 'word_counts.pickle', 'wb') as handle:
            pickle.dump(self.token.word_counts, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(save_path + 'word_docs.pickle', 'wb') as handle:
            pickle.dump(self.token.word_docs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(save_path + 'word_index.pickle', 'wb') as handle:
            pickle.dump(self.token.word_index, handle, protocol=pickle.HIGHEST_PROTOCOL)