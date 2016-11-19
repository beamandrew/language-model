import numpy as np


import tensorflow as tf
from data_utils import Vocabulary, Dataset

vocab = Vocabulary.from_file("1b_word_vocab.txt")
print("loaded vocab, num tokens " + str(vocab._num_tokens))

print("Loading data...")
dataset = Dataset(vocab, "/mnt/raid1/billion-word-corpus/1-billion-word-language-modeling-benchmark/training-monolingual.tokenized.shuffled/*")
print("Data loaded")

it = dataset.iterate_once(128,20)
for i, (x, y, w) in enumerate(it):
    print(str(x))