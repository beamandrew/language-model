import os
import string
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

class Dataset(object):
    def __init__(self,data_dir,num_words=None):
        self.data_dir = data_dir
        self.file_list = os.listdir(data_dir)
        self.all_texts = []
        for f in self.file_list[0:10]:
            print f
            reader = open(os.path.join(data_dir,f))
            txt = reader.read()
            txt = txt.split('\n')
            self.all_texts.extend(txt)
            reader.close()
        print(str(len(self.all_texts)))
        self.token = Tokenizer(nb_words=num_words, lower=True, split=' ')
        self.token.fit_on_texts(self.all_texts)
        self.vocab_size = len(self.token.word_index)
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
    def Y_to_Categorical(self,Y):
        num_classes = self.vocab_size
        Y_cat = np.zeros((Y.shape[0], len(Y[0]), num_classes))
        for i in range(len(Y)):
            Y_cat[i]  = to_categorical(Y[i], nb_classes=num_classes)
        return Y_cat