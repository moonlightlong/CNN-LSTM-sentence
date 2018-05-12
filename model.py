#!/usr/bin/python
# -*- coding:utf-8 -*-
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
import chnsenticorp
import data_helpers
from w2v import train_word2vec

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import MaxPooling1D, Convolution1D, Embedding
from keras.layers import LSTM, Bidirectional
from keras.layers.merge import Concatenate
from keras.datasets import imdb
from keras.preprocessing import sequence

import keras.backend as K

# model
class LSTM_CNN_Model(object):

    def __init__(self, conf):
        
        # 选择的模型 rand|static|non-static
        self.model_type = conf.model_type

        # 选择数据集 IMDB|MR|CHNSENTICORP
        self.data_source = conf.data_source

        # 设在模型的超参数
        self.embedding_dim = conf.embedding_dim
        self.filter_sizes = (3, 8)
        self.num_filters = conf.num_filters
        self.dropout_prob = (0.5, 0.8)
        self.hidden_dims = 50

        # 设置训练参数
        self.batch_size = conf.batch_size
        self.num_epochs = conf.num_epochs
        self.flags = False

        # 设置预处理参数
        self.sequence_length = conf.sequence_length
        self.max_words = 5000

        # 设置 word2vec 参数
        self.min_word_count = 1
        self.context = 10

        # 设置LSTM 输出dim
        self.lstm_output_dim = conf.lstm_output_dim

        # 读取数据
        print("=" * 120)
        print("Load data...")
        self.x_train, self.y_train, self.x_dev, self.y_dev, self.x_test, self.y_test, self.vocabulary_inv = self.load_data()
        
        if self.sequence_length != self.x_test.shape[1]:
            print("Adjusting sequence length for actual size")
            self.sequence_length = self.x_test.shape[1]

        print("x_train shape:", self.x_train.shape)
        print("x_test shape:", self.x_test.shape)
        print("Vocabulary Size: {:d}".format(len(self.vocabulary_inv)))
        print("max words:{}".format(self.max_words))
        print("=" * 120)
        print()
        self.build_model()

    def load_data(self):
        """读取数据
        """
        np.random.seed(0)

        assert self.data_source in ["IMDB", "MR", "CHNSENTICORP"], "Unknown data source"
        if self.data_source == "IMDB":
            (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=self.max_words, start_char=None,
                                                                oov_char=None, index_from=None)

            x_train = sequence.pad_sequences(x_train, maxlen=self.sequence_length, padding="post", truncating="post")
            x_test = sequence.pad_sequences(x_test, maxlen=self.sequence_length, padding="post", truncating="post")

            vocabulary = imdb.get_word_index()
            vocabulary_inv = dict((v, k) for k, v in vocabulary.items())
            vocabulary_inv[0] = "<PAD/>"
        else:
            if self.data_source == "MR":
                x, y, vocabulary, vocabulary_inv_list = data_helpers.load_data()
            else:
                x, y, vocabulary, vocabulary_inv_list = chnsenticorp.load_data()
            vocabulary_inv = {key: value for key, value in enumerate(vocabulary_inv_list)}
            y = y.argmax(axis=1)

            # Shuffle data
            shuffle_indices = np.random.permutation(np.arange(len(y)))
            x = x[shuffle_indices]
            y = y[shuffle_indices]
            train_len = int(len(x) * 0.7)
            x_train = x[:train_len]
            y_train = y[:train_len]
            x_test = x[train_len:]
            y_test = y[train_len:]
        shuffle_ind = np.random.permutation(np.arange(len(y_test)))
        x_test = x_test[shuffle_ind]
        y_test = y_test[shuffle_ind]
        test_len = int(len(y_test) * 0.5)
        x_dev = x_test[:test_len]
        y_dev = y_test[:test_len]
        x_test = x_test[test_len:]
        y_test = y_test[test_len:]

        return x_train, y_train, x_dev, y_dev, x_test, y_test, vocabulary_inv

    def build_model(self, flags=False):
        print("=" * 120)
        print("Model type is", self.model_type)
        if self.model_type in ["non-static", "static"]:
            embedding_weights = train_word2vec(np.vstack((self.x_train, self.x_test)), self.vocabulary_inv, num_features=self.embedding_dim,
                                            min_word_count=self.min_word_count, context=self.context)
            if self.model_type == "static":
                self.x_train = np.stack([np.stack([embedding_weights[word] for word in sentence]) for sentence in self.x_train])
                self.x_test = np.stack([np.stack([embedding_weights[word] for word in sentence]) for sentence in self.x_test])
                self.x_dev = np.stack([np.stack([embedding_weights[word] for word in sentence]) for sentence in self.x_dev])
                print("x_train static shape:", self.x_train.shape)
                print("x_test static shape:", self.x_test.shape)
                print("x_test static shape:", self.x_dev.shape)

        elif self.model_type == "rand":
            embedding_weights = None
        else:
            raise ValueError("Unknown model type")

        # 构建模型
        if self.model_type == "static":
            input_shape = (self.sequence_length, self.embedding_dim)
        else:
            input_shape = (self.sequence_length,)

        model_input = Input(shape=input_shape)

        # 静态模型， 没有embedding 层
        if self.model_type == "static":
            z = model_input
        else:
            z = Embedding(len(self.vocabulary_inv), self.embedding_dim, input_length=self.sequence_length, name="embedding")(model_input)

        z = Dropout(self.dropout_prob[0])(z)

        # 卷积模块
        conv_blocks = []
        # 双向LSTM模块
        lstm_blocks = []
        for sz in self.filter_sizes:
            convo = Convolution1D(filters=self.num_filters,
                                kernel_size=sz,
                                padding="valid",
                                activation="relu",
                                strides=1)(z)
            conv = MaxPooling1D(pool_size=2)(convo)
            conv_lstm = MaxPooling1D(pool_size=4)(convo)
            lstm = Bidirectional(LSTM(self.hidden_dims),
                                 input_shape=(int((self.sequence_length-sz+1)/4), self.num_filters))(conv_lstm)
            conv = Flatten()(conv)
            conv_blocks.append(conv)

            lstm_blocks.append(lstm)
        z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
        l = Concatenate()(lstm_blocks) if len(lstm_blocks) > 1 else lstm_blocks[0]

        z = Dropout(self.dropout_prob[1])(z)
        z = Dense(self.hidden_dims, activation="relu")(z)
        z = Concatenate()([z, l])
        model_output = Dense(1, activation="sigmoid")(z)

        self.model = Model(model_input, model_output)

        def recall(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

            recall = true_positives / (possible_positives + K.epsilon())
            return recall

        def precision(y_true, y_pred):
            true_posotives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_posotives / (predicted_positives + K.epsilon())
            return precision

        def f1(y_true, y_pred):
            pre = precision(y_true, y_pred)
            rec = recall(y_true, y_pred)
            return 2 * ((pre * rec)/(pre+rec+K.epsilon()))

        if flags == False:
            self.model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        else:
            self.flags = True
            self.model.compile(loss="binary_crossentropy",
                               optimizer="adam",
                               metrics=[precision, recall, f1])

        # 使用word2vec 初始化 权值
        if self.model_type == "non-static":
            weights = np.array([v for v in embedding_weights.values()])
            print("Initializing embedding layer with word2vec weights, shape", weights.shape)
            embedding_layer = self.model.get_layer("embedding")
            embedding_layer.set_weights([weights])    

    def fit(self):
        """训练模型
        """
        print("=" * 120)
        if self.flags == False:
            history = self.model.fit(self.x_train,
                                     self.y_train,
                                     batch_size=self.batch_size,
                                     epochs=self.num_epochs,
                                     validation_data=(self.x_dev, self.y_dev))
        else:
            history = self.model.fit(self.x_train,
                                     self.y_train,
                                     batch_size=self.batch_size,
                                     epochs=self.num_epochs,
                                     validation_data=(self.x_dev, self.y_dev))
        return history
            

    def evaluate(self):
        """预测，获得准确度
        """
        print("=" * 120)
        score, acc = self.model.evaluate(self.x_test, self.y_test, batch_size=self.batch_size)
        print("Test score:", score)
        print("Test accuracy:", acc)
