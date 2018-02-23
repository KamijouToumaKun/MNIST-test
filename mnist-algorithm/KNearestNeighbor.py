# -*- coding: UTF-8 -*-

import numpy as np
from numpy import matrix
import math
import os
import operator

class KNearestNeighbor:
    TRAIN_DATA_SIZE = 1000

    def __init__(self, _k, data_matrix, data_labels, train_indice):
        self._k = _k
        self.data_matrix = []
        self.data_labels = []
        #内存不够训练60000个数据，够训练10000个但是太慢，这里只训练随机选取的1000个
        for i in range(self.TRAIN_DATA_SIZE):
            self.data_matrix.append(data_matrix[train_indice[i]])
            self.data_labels.append(data_labels[train_indice[i]])

    def partition(self, ilist, alist, start, end):
        if end <= start:
            return
        key, value = ilist[start], alist[start]
        while start < end:
            while start < end and alist[end] >= value:
                end -= 1
            ilist[start], alist[start] = ilist[end], alist[end]
            while start < end and alist[start] <= value:
                start += 1
            ilist[end], alist[end] = ilist[start], alist[start]
        ilist[start], alist[start] = key, value
        return start #start==end

    def find_least_k_indexs(self, ilist, alist, k):
        length = len(alist)
        #if length == k:
        #    return alist
        # if not alist or k <=0 or k > length:
            # return []
        start, end = 0, length-1
        index = self.partition(ilist, alist, start, end)
        while index != k:
            if index > k:
                index = self.partition(ilist, alist, start, index-1)
            elif index < k:
                index = self.partition(ilist, alist, index+1, end)
        return ilist[:k]

    #no training
    def predict(self, test):
        #计算测试数据到所有训练数据的欧氏距离
        diff_mat = np.tile(test, (np.shape(self.data_matrix)[0], 1)) - self.data_matrix  
        #diff_mat = np.array(diff_mat)  
        square_diff_mat = diff_mat ** 2  
        square_distances = square_diff_mat.sum(axis=1)  

        # 这里尝试用快排的方式找前k小的距离，但是因为1000的数据量本来就很小，因此节约不了时间
        # classCount = {} 
        # min_k_indexs = self.find_least_k_indexs(range(self.TRAIN_DATA_SIZE), square_distances, self._k)
        # for index in min_k_indexs:
        #     voteLabel = self.data_labels[index]  
        #     classCount[voteLabel] = classCount.get(voteLabel,0) + 1  
        # sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1),reverse=True)  
        # return sortedClassCount[0][0]  

        # 排序找前k小的距离
        sortedDistanceIndex = square_distances.argsort()  
        classCount = {} 
        # 前k近的点投票决定测试数据的归属
        for i in range(self._k):  
            voteLabel = self.data_labels[sortedDistanceIndex[i]]  
            classCount[voteLabel] = classCount.get(voteLabel,0) + 1  
        sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1),reverse=True)  
        return sortedClassCount[0][0]  