# -*- coding: UTF-8 -*-

import numpy as np
from numpy import matrix
from collections import namedtuple
import math
import os
import operator
import json

class NaiveBayes:
    #参数的保存路径
    NN_FILE_PATH = 'NaiveBayes.json'

    def __init__(self, data_matrix, data_labels, training_indices, use_file=True):
        # 决定了要不要导入json
        self._use_file = use_file
        #数据集
        self.data_matrix = data_matrix
        self.data_labels = data_labels
        #状态储存，离散化地认为一个像素点只有两种状态：是否>=0.5
        self.count = [[[-1 for y0i in range(2)] for i in range(28*28)] for label in range(10)]

        if (not os.path.isfile(NaiveBayes.NN_FILE_PATH) or not use_file):
            #训练并保存
            TrainData = namedtuple('TrainData', ['y0', 'label'])
            self.train([TrainData(self.data_matrix[i], int(self.data_labels[i])) for i in training_indices])
            self.save()
        else:
            # 如果json存在则加载
            self._load()

        #每个label的data个数、总data个数
        self.c_label = [np.sum(self.count[label][0]) for label in range(10)]
        self.c_total = np.sum(self.c_label)

    def train(self, training_data_array):
        dict_data = [[] for x in range(10)]
        for data in training_data_array:
            dict_data[data.label].append(data.y0)
        #求count[label][i][y0i]，即：其label为label 且 其y0[i] = y0i的data数量
        for label in range(10):
            for i in range(28*28):
                for y0i in range(2):
                    c_label_y0i = 0
                    for dict_y0 in dict_data[label]:
                        dict_state = 1 if dict_y0[i] >= 0.5 else 0
                        if dict_state == y0i:
                            c_label_y0i += 1
                    self.count[label][i][y0i] = c_label_y0i

    #给定y0与label，求P(y0|label) P(label)的对数
    def log_p_y0_label(self, y0, label):
        #由朴素贝叶斯假设
        #即求P(y0[0]|label) P(y0[1]|label) ... P(y0[N-1]|label) P(label)的对数
        #求P(label)，这里默认len(self.dict[label]) > 0
        log_p = math.log(self.c_label[label]) - math.log(self.c_total)
        for i in range(28*28):
            #求P(y0[i]|label)
            y0i = 1 if y0[i]>=0.5 else 0
            c_label_y0i = self.count[label][i][y0i]
            c_label = self.c_label[label]
            #拉普拉斯平滑
            if c_label_y0i == 0:
                c_label_y0i = 1
                c_label += 2 #特征有2个：y0[i] = 1或0
            log_p += math.log(c_label_y0i) - math.log(c_label)
        return log_p

    #给定y0，求argmax P(label|y0)
    #只需求argmax P(y0|label) P(label)
    def predict(self, test):
        results = [self.log_p_y0_label(test, label) for label in range(10)]
        return results.index(max(results))

    def save(self):
        if not self._use_file:
            return

        json_neural_network = {
            "count": self.count
        };
        with open(NaiveBayes.NN_FILE_PATH,'w') as nnFile:
            json.dump(json_neural_network, nnFile)

    def _load(self):
        if not self._use_file:
            return

        with open(NaiveBayes.NN_FILE_PATH) as nnFile:
            nn = json.load(nnFile)
        self.count = nn['count']
