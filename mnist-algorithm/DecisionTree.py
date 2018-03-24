# -*- coding: UTF-8 -*-

# 普通的决策树是二分类
# 普通决策树的每一个节点都只看一个自变量/特征进行决策，则自变量取值是离散化的
# 就相当于用一条平行于“该变量对应的坐标轴”的线段对部分样本点进行分类
# 许多这样的线段连在一起，成为折线，对所有样本点进行分类

# 特殊的决策树可以多分类
# 也可以每一个节点都看多个自变量的加权来进行决策，自变量的取值可以连续化了
# 就相当于用一条斜线段对部分样本点进行分类

import os
import sys
import math
import json
import operator

class DecisionTree:
    #参数的保存路径；保存一棵树需要递归，暂时懒得写了……
    # NN_FILE_PATH = 'DecisionTree.json'

    def __init__(self, data_matrix, data_labels, training_indices, use_file=True):
        # 决定了要不要导入json
        self._use_file = use_file
        # 数据集
        self.data_matrix = []
        # 状态离散化处理
        for data in data_matrix:
            # data不能直接作为左值
            self.data_matrix.append([1 if x>=0.5 else 0 for x in data])
        self.data_labels = data_labels

        if (not os.path.isfile(DecisionTree.NN_FILE_PATH) or not use_file):
            #训练并保存
            self.train(training_indices)
            # self.save()
        # else:
            # 如果json存在则加载
            # self._load()

    #求信息熵 E(S) = ∑p(label) * logp(label)；不需要知道当前的数据特征
    def calcShannonEnt(self, data_indexs):  
        labelCounts = {}  
        number = 0  
        for i in data_indexs:
            currentLabel = self.data_labels[i]  
            if currentLabel not in labelCounts.keys():  
                labelCounts[currentLabel] = 0  
            labelCounts[currentLabel] += 1  
            number += 1  
        shannonEnt = 0.0  
        for key in labelCounts:  
            prob = float(labelCounts[key])/len(data_indexs)
            shannonEnt -= prob * math.log(prob,2)  
        return shannonEnt  

    #利用信息增益来决定该如何对数据集进行划分：
    def chooseBestFeatureToSplit(self, data_indexs, feat_indexs):  
        baseEntropy = self.calcShannonEnt(data_indexs)

        _max = -sys.maxint-1
        _argmax = -1
        for j in feat_indexs:  #对每一维特征
            featList = []
            for i in data_indexs:
                featList.append(self.data_matrix[i][j]) #取得这一子列
            uniqueVals = set(featList)  
            
            newEntropy = 0.0  
            # sub_feat_indexs = feat_indexs.remove(j) #删除这一子列
            for value in uniqueVals:  #每一行根据这一子列的不同取值
                sub_data_indexs = self.splitDataSet(data_indexs, j, value) #把数据集按行划分
                prob = len(sub_data_indexs) / float(len(data_indexs)) #计算子集的占比
                # E(S, A) = ∑p(Sv) * E(Sv)
                newEntropy += prob * self.calcShannonEnt(sub_data_indexs)          
            # Gain(S, A) = E(S)–E(S, A)
            infoGain = baseEntropy - newEntropy  
            if infoGain > _max:
                _max = infoGain
                _argmax = j

        return _argmax #计算哪一维的增益最大

    def splitDataSet(self, data_indexs, axis, value): 
        #数据集只保留axis这一维特征的值为value的元组，但是删去这些元组的这一维
        sub_data_indexs = set() 
        for i in data_indexs:  
            if self.data_matrix[i][axis] == value:
                sub_data_indexs.add(i)  
        return sub_data_indexs  

    def majorityCnt(self, labels):  #返回分类结果中的多数
        labelCount = {}  
        for vote in labels:  
            if vote not in labelCount.keys():  
                labelCount[vote] = 0  
            labelCount[vote] += 1  
        sortedLabelCount = sorted(labelCount.iteritems(), key=operator.itemgetter(1), reverse=True)  
        return sortedLabelCount[0][0] 

    def createTree(self, data_indexs, feat_indexs):  
        # data_indexs表示当前数据子集在数据库中的行号
        # feat_indexs表示当前数据子集在数据库中的列号

        # 表示当前data最后可能对应到哪些label；这里可以优化
        labels = [self.data_labels[i] for i in data_indexs]  
        if labels.count(labels[0]) == len(labels):  #如果样本全是一类，就作为叶节点，直接返回
            return labels[0]  
        if len(feat_indexs) == 0:  #如果不存在特征了，就作为叶节点，直接返回labels的多数
            return self.majorityCnt(labels) 

        bestFeatIndex = self.chooseBestFeatureToSplit(data_indexs, feat_indexs)  
        myTree = {bestFeatIndex:{}}  
        feat_indexs.remove(bestFeatIndex)  #删去这一维特征

        featList = []
        for i in data_indexs:
            featList.append(self.data_matrix[i][bestFeatIndex]) #取得这一子列
        uniqueVals = set(featList)  
        for value in uniqueVals:  #对于这一特征的所有取值
            sub_feat_indexs = feat_indexs #这一子列已经删除
            # 数据集删去这一维特征，按照这一维的value取值不同，递归划分出各个子节点
            myTree[bestFeatIndex][value] = self.createTree(
                self.splitDataSet(data_indexs, bestFeatIndex, value), 
                sub_feat_indexs)
        
        return myTree  

    def train(self, training_indices):
        self.tree = self.createTree(set(training_indices), set(range(28*28)))  
    
    def predict(self, test):
        # 离散化
        test = [1 if x>=0.5 else 0 for x in test]
        node = self.tree
        while type(node) == dict:
            try:
                featIndex = node.keys()[0]
                node = node[featIndex] #恰有一个最佳特征，选择这个特征
                if node.has_key(test[featIndex]):
                    node = node[test[featIndex]] #根据这个特征的值选择分支
                else: #*如果训练时因为数据稀疏，没有遇到这个特征这样取值的情况
                    node = node[node.keys()[0]] #那就随便走一个分支了
            except Exception as e:
                print node
            else:
                pass
        return node

    def save(self):
        if not self._use_file:
            return

        json_neural_network = {
            "tree": self.tree
        };
        with open(DecisionTree.NN_FILE_PATH,'w') as nnFile:
            json.dump(json_neural_network, nnFile)

    def _load(self):
        if not self._use_file:
            return

        with open(DecisionTree.NN_FILE_PATH) as nnFile:
            nn = json.load(nnFile)
        self.tree = nn['tree']