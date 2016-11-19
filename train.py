import numpy as np


import tensorflow as tf
from data_utils import Vocabulary, Dataset

pvocab = Vocabulary.from_file("1b_word_vocab.txt")
print("loaded vocab")