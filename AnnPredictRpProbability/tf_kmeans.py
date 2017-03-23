#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf




class TFKMeans(object):
    '''
    KMeansCluster In TensorFlow
    '''

    def __init__(self,k=10,max_iter=300, session=None):
        '''
        _k: 聚类个数
        _max_iter:最大迭代次数
        '''
        self._k = k
        self._max_iter = max_iter
        self.sess = session or tf.Session()

    def _randomInitCenter(self, dataX):
        '''
        随机初始化节点
        '''
        n_samples = tf.shape(dataX)[0]
        random_indices = tf.random_shuffle(tf.range(0, n_samples))

        begin = [0,]
        size = [self._k,]
        # 切片: 选择打乱数据的前k个
        center_indices = tf.slice(random_indices, begin, size)
        centers = tf.gather(dataX, center_indices)
        return centers

    def updateCluster(self, dataX, centers):
        '''
        更新簇
        '''
        # tf.substract的输入要求扩展维
        expanded_data = tf.expand_dims(dataX,0)
        expanded_center = tf.expand_dims(centers, 1)
        distance = tf.reduce_sum(tf.square(
                       tf.sub(expanded_data, expanded_center)
                    ), 2)

        # 找到每个点最近的中心
        near_indices = tf.argmin(distance,0)
        # 当前的损失
        loss = tf.reduce_mean(tf.reduce_min(distance, 0))
        return near_indices, loss

    def updateCenter(self, dataX, nearest):
        '''
        更新质心
        '''

        # dynamic_partition: 分组
        partitions = tf.dynamic_partition(dataX, tf.to_int32(nearest), self._k)

        #print self.sess.run(partitions)
        # 平均值更新centers
        new_centers = tf.concat([tf.expand_dims(tf.reduce_mean(partition, 0), 0) for partition in partitions], 0)
        #new_centers = tf.constant([tf.reduce_mean(p, 0) for p in partitions])
        return new_centers


    def train(self, dataX):
        '''
        训练
        数据要求: np.ndarry 或者 tf
        '''
        #try:
            #dataX = self.sess.run(dataX)
        #except:
        dataX = self.sess.run(tf.constant(dataX))

        initcenters = self._randomInitCenter(dataX)
        #print dataX
        #print self.sess.run(tf.shape(dataX))
        #print self.sess.run(tf.shape(tf.constant([[1,2,3],[3,4,5]])))
        # 迭代第一次
        nearest, loss = self.updateCluster(dataX, initcenters)
        centers =  self.sess.run(self.updateCenter(dataX, nearest))
        lastloss = self.sess.run(loss)

        for i in range(self._max_iter):
            print 'iter:',i, 'loss',

            # 交替更新nearest_indices(簇)以及重心
            nearest_indices, loss=self.updateCluster(dataX, centers)
            centers = self.sess.run(self.updateCenter(dataX, nearest_indices))
            lossvalue = self.sess.run(loss)
            print lossvalue
            if lastloss - lossvalue < 0.0001:
                print 'finsied'
                break
            lastloss = lossvalue

        self.centers = centers
