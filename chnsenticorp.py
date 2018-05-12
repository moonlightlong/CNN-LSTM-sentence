#!/usr/bin/python
# -*- coding:utf-8 -*-
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os

import numpy as np
import re
import itertools
from collections import Counter

import codecs

"""
Original taken from https://github.com/dennybritz/cnn-text-classification-tf
"""

def clean_s(string):
    
    string = re.sub(r"\/[a-zA-Z]{1,}", " ", string)
    
    string = re.sub(r"[^A-Za-z0-9\u4e00-\u9fa5(),!?\'\`]", " ", string) 
    string = re.sub(r",", "，", string)
    string = re.sub(r"\.", "。", string)
    string = re.sub(r":", "：", string)
    string = re.sub(r";", "；", string)
    string = re.sub(r"!", "！", string)
    string = re.sub(r"\?", "\？", string)
    string = re.sub(r"\(", "\（", string)
    string = re.sub(r"\)", "\）", string)
    string = re.sub(r"", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string

def load_data_and_labels():
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    path = "./data/ChnSentiCorp-htl/6000/"

    neg = path + "neg"
    pos = path + "pos"
    simples_neg = []
    for file in os.listdir(neg):
        simple = ""
        for l in codecs.open(neg + "/" + file, "rb", "utf-8").readlines():
            if len(l) != 0:
                simple += l.strip('\n')
        simples_neg.append(simple)
    simples_pos = []
    for file in os.listdir(pos):
        simple = ""
        for l in codecs.open(pos + "/" + file, "rb", "utf-8").readlines():
            if len(l) != 0:
                simple += l.strip('\n')
        simples_pos.append(simple)
    
    x_text = simples_neg + simples_pos
    x_text = [clean_s(sent) for sent in x_text]
    x_text = [s.split(" ") for s in x_text]

    neg_labels = [[0, 1] for _ in simples_neg]
    pos_labels = [[1, 0] for _ in simples_pos]
    y = np.concatenate([neg_labels, pos_labels], 0)
    return [x_text, y]

def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]

def load_data():
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels()
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]


def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

