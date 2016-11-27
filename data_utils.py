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
        self.progbar = generic_utils.Progbar(len(file_list))
    def __iter__(self):
        for fname in self.file_list:
            self.progbar.add(1)
            with open(os.path.join(self.data_dir,fname)) as f:
                for line in f:
                    yield line


class Dataset(object):
    def __init__(self,data_dir,num_words=None):
        self.texts = DataGenerator(data_dir)
        self.token = Tokenizer(nb_words=num_words, lower=True, split=' ')
        print('Reading files...')
        self.token.fit_on_texts(self.texts)
        self.vocab_size = len(self.token.word_index)
        print(str(self.vocab_size))
        if num_words is not None:
            self.vocab_size = num_words
    def create_X_Y(self,seq_len=25,one_hot_y=False):
        proc_txt = self.token.texts_to_sequences(self.all_texts)
        X = [txt[:min(len(txt),seq_len)] for txt in proc_txt]
        X = pad_sequences(X)
        if(one_hot_y):
            Y = []
            num_classes =  self.vocab_size
            for txt in proc_txt:
                y_txt = txt[1:min(len(txt),seq_len)+1]
                Y.append(to_categorical(y_txt, nb_classes=num_classes).tolist())
        else:
            Y = [txt[1:min(len(txt),seq_len)+1] for txt in proc_txt]
            Y = pad_sequences(Y)
        return X,Y