import os
import string
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.utils import generic_utils

class DataGenerator(object):
    def __init__(self,data_dir):
        self.file_list = os.listdir(data_dir)
        self.data_dir = data_dir
        self.progbar = generic_utils.Progbar(len(self.file_list))
    def __iter__(self):
        for fname in self.file_list[0:2]:
            self.progbar.add(1)
            with open(os.path.join(self.data_dir,fname)) as f:
                for line in f:
                    yield line


class Dataset(object):
    def __init__(self,data_dir,num_words=None):
        self.file_list = os.listdir(data_dir)
        self.data_dir = data_dir
        self.texts = DataGenerator(data_dir)
        self.token = Tokenizer(nb_words=num_words, lower=True, split=' ')
        print('Reading files...')
        self.token.fit_on_texts(self.texts)
        self.vocab_size = len(self.token.word_index)
        print(str(self.vocab_size))
        if num_words is not None:
            self.vocab_size = num_words
        self.batch_size = 32
        self.seq_len = 25
    def  __iter__(self):
        for fname in self.file_list:
            with open(os.path.join(self.data_dir, fname)) as f:
                lines = []
                for line in f:
                    if len(lines) == self.batch_size:
                        proc_txt = self.token.texts_to_sequences(lines)
                        X_batch = [txt[:min(len(txt)-1, self.seq_len)] for txt in proc_txt]
                        X_batch = pad_sequences(X_batch)
                        Y_batch = [txt[1:min(len(txt), self.seq_len) + 1] for txt in proc_txt]
                        Y_batch = pad_sequences(Y_batch)
                        lines = []
                        yield X_batch,Y_batch
                    else:
                        lines.append(line)
    def set_batch_size(self,batch_size):
        self.batch_size = batch_size
    def set_data_dir(self,data_dir):
        self.data_dir = data_dir
        self.file_list = os.listdir(data_dir)
