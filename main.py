#!/usr/bin/python
# -*- coding:utf-8 -*-
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
# if tensorflow is imported, please do as following:
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import model
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="LSTM_CNN model")

# 选择的模型 rand|static|non-static
parser.add_argument('--model_type', type=str, default='rand', help='rand|static|non-static')

# 选择数据集 IMDB|MR|CHNSENTICORP
parser.add_argument('--data_source', type=str, default='IMDB', help='IMDB|MR|CHNSENTICORP')

# 设在模型的超参数
parser.add_argument('--embedding_dim', type=int, default=50, help='the word embedding dim')
parser.add_argument('--num_filters', type=int, default=10)
parser.add_argument('--hidden_dims', type=int, default=50)

# 设置训练参数
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_epochs', type=int, default=10)

# 设置预处理参数
parser.add_argument('--sequence_length', type=int, default=400)
parser.add_argument('--max_words', type=int, default=5000)

# 设置 word2vec 参数
parser.add_argument('--mun_word_count', type=int, default=1)
parser.add_argument('--context', type=int, default=10)

# 设置LSTM 输出dim
parser.add_argument('--lstm_output_dim', type=int, default=50)

# 输出已设置的参数
conf = parser.parse_args()

# #########################################################################
# 训练
# #########################################################################
data_sets = ["MR", "CHNSENTICORP"]
history_dict = {}
metrics_dict = {}
for data_name in data_sets:
    conf.data_source = data_name
    model_type_list = ["rand", "static", "non-static"]
    for model_type in model_type_list:
        conf.model_type = model_type

        # if model_type == "rand":
        #     os.remove('./models/50features_1minwords_10context')
        
        # 输出参数
        print("")
        print("#" * 210)
        print(data_name + model_type)
        print("Parameters:")
        print(conf)
        print("")

        lcmodel = model.LSTM_CNN_Model(conf)
        imdb_h, metrics = lcmodel.fit()
        history_dict[data_name + model_type] = imdb_h
        metrics_dict[data_name + model_type] = metrics

        if data_name + model_type == "MRrand":
            from keras.utils import plot_model
            plot_model(lcmodel.model, to_file='model.png')
        
        if data_name + model_type in ["MRnon-static", "CHNSENTICORPnon-static"]:
            print("")
            print(lcmodel.model.summary())

for f in metrics_dict.keys().sort():
    print(f + ":")
    print("precision:" + max(metrics_dict[f].val_precisions))

import matplotlib.pyplot as plt

filename = []
for file in history_dict.keys().sort():
    plt.plot(history_dict[file].history["val_acc"])
    filename.append(file)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(filename, loc='upper left')
plt.show()