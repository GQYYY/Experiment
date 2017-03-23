#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import copy
import apRssiStatistics

class HammingKMeans(object):

    def __init__(self,k,apFingerprints,rpCoordinates,apList,rpList,max_iter=300):
        '''
        k: 聚类的数目
        max_iter: 最大迭代次数
        '''
        self._k = k
        self._max_iter = max_iter
        self._centers = None #各个聚类的中心指纹
        self._init_centers = None #初始化的聚类中心，随机选择的指纹
        #各指纹的聚类信息
        #第一列存储当前指纹所属的聚类
        #第二列存储当前指纹与聚类中心的Hamming Distance
        self._clusterAssment = None
        self._label = None #各指纹所属的聚类

        self._apFinerprints = apFingerprints
        self._rpCoordinates = rpCoordinates
        self._apList = apList
        self._rpList = rpList

        s = apRssStatistics(apFingerprints,rpCoordinates,apList,rpList)
        self._ri = s.get_ri()      #reliability indicator
        self._delta = s.get_delta  #Variance of readings for all RPs


    #计算RP的Hamming Distaance
    def _calHammingDistance(self, rp1, rp2):
        index_rp1 = self._rpList.index(rp1)
        index_rp2 = self._rpList.index(rp2)
        hamming_distance = np.sum(np.abs(self._ri[index_rp1,:] - self._ri[index_rp2,:]))
        return hamming_distance

    #随机选择k个节点(RP)作为聚类中心
    def _randomInitCenter(self):  
        shuffle_indices = np.random.permutation(len(self._rpList))
        centers = shuffle_indices[:k] #center中记录着随机选取的RP在rpList中的索引
        return centers

    def train(self):
        '''
        训练数据
        '''
        if not isinstance(apFingerprints, np.ndarray):
            try:
                apFingerprints = np.asarray(apFingerprints)
            except:
                raise TypeError('np.ndarray is necessary')

        self._centers = self._randomInitCenter(apFingerprints, self._k)
        self._init_centers= copy.deepcopy(self._centers)

        Num = len(self._rpList)
        # 第一列存储当前RP属于的簇
        # 第二列存储当前RP据与簇中心的Hamming距离(err)
        self._clusterAssment = np.empty((Num,2))

        for i in xrange(self._max_iter):
            isConverge = True
            # step1: 更新簇内元素
            print 'iter:', i, 'starts',
            for i in range(Num):
                MinDist = np.inf
                ind = 0
                for j in range(self._k):
                    # 寻找最近的中心
                    dist = self._calHammingDistance(self._apList[i], self._apList(self._centers[j]))
                    if dist < MinDist:
                        MinDist = dist
                        ind = j
                if self._clusterAssment[i,0] != ind:
                    isConverge = False
                    self._clusterAssment[i,:] = ind, MinDist

            # 所有数据均未修改所属簇，已经收敛
            if isConverge:
                break

            # step2: 更新质心
            for i in range(self._k):
                allindex = self._clusterAssment[:,0]  #所有RP对应的簇号（0...k-1）
                cur_index = np.nonzero(allindex == i)[0] #返回allindex中为i的(即属于簇i的)在allindex中的下标索引
                if len(self._rpList[cur_index]) == 0:
                    continue
                #将该簇内平均方差最小的RP设为该簇的质心
                self._centers[i] = np.argmin(self._delta[cur_index])
            print np.sum(self._clusterAssment[:,1])
        self._label = self._clusterAssment[:, 0]
        self._sse = sum(self._clusterAssment[:,1])

    #根据train结果进行预测
    def predict(self, X):
        if not isinstance(X, np.ndarray):
            try:
                X = np.asarray(X)
            except:
                raise TypeError('np.ndarray is necessary')

        num = X.shape[0]
        result = np.empty((num,))

        for i in range(num):
            MinDist = np.inf
            for j in range(self._k):
                dist = self._calDist(X[i,:], self._centers[j,:])
                if MinDist > dist:
                    MinDist = dist
                    index = j

            result[i] = index

        return result
