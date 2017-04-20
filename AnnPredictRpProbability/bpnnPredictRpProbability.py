#!/usr/bin/env python
# coding=utf-8

import numpy as np
import tensorflow as tf
import dataProcessor as dp
import time
import data_helper

L1_loss = 0.0 #L1正则项
L2_loss = 0.0 #L2正则项

def make_layer(inputs, input_size, output_size, activate=None,keep_prob=1.0, name='layer'):


        # weights = tf.Variable(tf.random_uniform([input_size, output_size],-1.0,1.0))
        with tf.variable_scope(name):
            weights = tf.get_variable('weight',shape=[input_size, output_size], initializer=tf.contrib.layers.xavier_initializer())

        global L1_loss
        global L2_loss

        L1_loss += tf.reduce_sum(tf.abs(weights))
        L2_loss += tf.nn.l2_loss(weights)

        #biases = tf.Variable(tf.zeros([1, output_size]) + 0.1)
        #返回一个shape为[1, output_size]的tensor，其中的元素服从minval和maxval之间的均匀分布
        #biases = tf.Variable(tf.random_uniform([1,output_size],minval=-1,maxval=1))
        biases = tf.Variable(tf.constant(0.1,shape=[1,output_size]))

        L1_loss += tf.reduce_sum(tf.abs(biases))
        L2_loss += tf.nn.l2_loss(biases)

        result = tf.matmul(inputs, weights) + biases
        result = tf.nn.dropout(result,keep_prob)

        if activate is None:
            return result
        else:
            return activate(result)

class bpnnPredictRpProbability:
    def __init__(self,train_inputs,train_outputs,test_inputs,test_outputs):
        self.session = tf.Session()
        self.loss = None
        self.optimizer = None
        self.input_n = 0  #输入层的维数，即输入层节点的个数
        self.hidden_n=0
        self.hidden_size = [] #列表:各隐层的节点个数
        self.output_n = 0  #输出层的维数，即输出层节点的个数
        self.input_layer = None
        self.hidden_layers = [] #列表:各隐层经激活函数处理之后的结果
        self.output_layer = None
        self.label_layer = None
        self.keep_prob = None

        self.train_inputs = train_inputs
        self.train_outputs = train_outputs
        self.test_inputs = test_inputs
        self.test_outputs = test_outputs

    def __del__(self):
        self.session.close()

    '''
    input_num:输入层节点个数
    hidden_num:隐层各层节点个数构成的列表
    output_num:输出层节点个数
    '''
    def setup(self, input_num, hidden_num, output_num):
        # set size args
        self.input_n = input_num #设置输入层节点个数
        self.hidden_n = len(hidden_num) #设置隐层层数
        self.hidden_size = hidden_num #用一个列表赋值，设置每个隐层的节点个数
        self.output_n = output_num #设置输出层节点个数

        # build input layer
        self.input_layer = tf.placeholder(tf.float32, [None, self.input_n])
        # build label layer
        self.label_layer = tf.placeholder(tf.float32, [None, self.output_n])
        self.keep_prob = tf.placeholder(tf.float32)

        # build hidden layers
        in_size = self.input_n
        out_size = self.hidden_size[0]
        inputs = self.input_layer


        self.hidden_layers.append(make_layer(inputs, in_size, out_size, activate=tf.nn.relu,keep_prob=self.keep_prob, name="input-layer"))
        for i in range(self.hidden_n-1):
            in_size = out_size
            out_size = self.hidden_size[i+1]
            inputs = self.hidden_layers[-1]
            self.hidden_layers.append(make_layer(inputs, in_size, out_size, activate=tf.nn.relu,keep_prob=self.keep_prob, name='hidden-layer-%s'%(str(i))))

        # build output layer
        #self.output_layer = make_layer(self.hidden_layers[-1], self.hidden_size[-1], self.output_n, activate=tf.nn.softmax,name='output')
        #测试tf.nn.sparse_softmax_cross_entropy_with_logits()函数，要求神经网络的输出层不经过softmax处理
        self.output_layer = make_layer(self.hidden_layers[-1], self.hidden_size[-1], self.output_n, activate=None,name='output')


    def trainAndTest(self,epochs=500,learn_rate=1e-3,keep_prob=0.5,batch_size=64,lambda_l1=0.0,lambda_l2=0.05):

        print ('train_inputs.shape:',self.train_inputs.shape)
        print ('train_outputs.shape:',self.train_outputs.shape)
        print ('test_inputs.shape:',self.test_inputs.shape)
        print ('test_outputs.shape:',self.test_outputs.shape)

        #使用cross_entropy函数作为loss函数
        #self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.output_layer, self.label_layer)) + lambda_l1 * L1_loss +lambda_l2 * L2_loss   #高版本tensorflow
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.output_layer,tf.argmax(self.label_layer,1))) + lambda_l1 * L1_loss +lambda_l2 * L2_loss #针对只有一个正确答案的分类更高效
        #self.loss = tf.reduce_mean(-tf.reduce_sum(self.label_layer * tf.log(self.output_layer))) + lambda_l1 * L1_loss +lambda_l2 * L2_loss                  #低版本tensorflow

        #self.optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(self.loss)
        self.optimizer = tf.train.AdamOptimizer(learn_rate).minimize(self.loss)

        #计算accuracy
        correct_prediction = tf.equal(tf.argmax(self.output_layer, 1), tf.argmax(self.label_layer, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 进行训练
        initer = tf.initialize_all_variables()
        self.session.run(initer)
        start_time = time.time()
        #write = tf.summary.FileWriter('logs/', self.session.graph)
        #write = tf.train.SummaryWriter('logs/', self.session.graph)

        for epoch in range(epochs):
            batches = data_helper.batch_iter(self.train_inputs, self.train_outputs, batch_size)
            for batch in batches:
                x,y = batch
                self.session.run(self.optimizer, feed_dict={self.input_layer: x, self.label_layer: y,self.keep_prob:keep_prob})

            print ('')
            print ('*'*30)
            #print 'output:'
            #print self.session.run(self.output_layer,feed_dict={self.input_layer:self.train_inputs,self.label_layer:self.train_outputs,self.keep_prob:1.0})
            print ('output argmax:')
            print (self.session.run(tf.argmax(self.output_layer,1),feed_dict = {self.input_layer:self.train_inputs,self.label_layer:self.train_outputs,self.keep_prob:1.0})[:50])
            print ('label argmax:')
            print (self.session.run(tf.argmax(self.label_layer,1),feed_dict={self.label_layer:self.train_outputs})[:50])

            #进行测试
            print ('epoch', epoch+1, 'loss:', self.session.run(self.loss, feed_dict={self.input_layer: self.train_inputs, self.label_layer: self.train_outputs,self.keep_prob:1.0}))
            print ('epoch', epoch+1, 'train accuracy:', self.session.run(accuracy, feed_dict={self.input_layer: self.train_inputs, self.label_layer: self.train_outputs,self.keep_prob:1.0}))
            print ('epoch', epoch+1, 'test accuracy:', self.session.run(accuracy,feed_dict={self.input_layer: self.test_inputs, self.label_layer:self.test_outputs,self.keep_prob:1.0}))
            print ('*'*30)
            print ('')

        end_time = time.time()
        print ('*' *60)
        print ('Training finish! Cost time:', int(end_time-start_time) , 'seconds')
        print ('Training accuracy:', self.session.run(accuracy, feed_dict={self.input_layer: self.train_inputs, self.label_layer: self.train_outputs,self.keep_prob:1.0}))
        print ('Testing accuracy:', self.session.run(accuracy, feed_dict={self.input_layer: self.test_inputs, self.label_layer: self.test_outputs,self.keep_prob:1.0}))
        print ('input dimension:',self.input_n)
        print ('hidden dimension:',self.hidden_size)
        print ('output dimension:',self.output_n)
        print ('learn_rate:',learn_rate)
        print ('keep_prob:',keep_prob)
        print ('lambda_l1:',lambda_l1)
        print ('lambda_l2:',lambda_l2)
        print ('batch_size:',batch_size)

def dataToOne_hotVector(_label,row,col):
    hotVectors = np.zeros([row,col])
    for i,label in enumerate(_label.tolist()):
        hotVectors[i,int(label)] = 1.0
    return hotVectors



if __name__ == '__main__':


    '''
    #两个参数分别为（训练数据文件名，测试数据文件名）
    dataProcessor = dp.dataProcessor(r'data4trainingNexus',r'bai4testing_1_4.log')
    #获得训练指纹(12400*92)
    trainingApFingerprints = dataProcessor.getTrainingApFingerprints() +100
    trainingCoordinates = dataProcessor.getTrainingCoordinates()
    #获得所有的位置列表
    coordinatesList = dataProcessor.getCoordinatesList()
    #获取和训练指纹所对应的RP的编号(12400*124)
    trainingCoordinatesId = dataProcessor.getTrainingCoordinatesId()
    #获取测试指纹
    testingApFingerprints = dataProcessor.getTestingApFingerprints() + 100
    #获取和训练指纹所对应的RP的编号
    testingCoordinatesId = dataProcessor.getTestingCoordinatesId()
    testingCoordinates = dataProcessor.getTestingCoordinates()

    np.save('./Data/trainingApFingerprints',trainingApFingerprints)
    np.save('./Data/trainingCoordinatesId',trainingCoordinatesId)
    np.save('./Data/trainingCoordinates',trainingCoordinates)
    np.save('./Data/testingApFingerprints',testingApFingerprints)
    np.save('./Data/testingCoordinatesId',testingCoordinatesId)
    np.save('./Data/testingCoordinates',testingCoordinates)
    np.save('./Data/coordinatesList',coordinatesList)
    print ('保存完毕')
    '''

    dataProcessor = dp.dataProcessor(r'data4trainingNexus',r'bai4testing_1_4.log')
    trainingApFingerprints = dataProcessor.getTrainingApFingerprints()
    trainingCoordinatesId = dataProcessor.getTrainingCoordinatesId()

    originalTrainingApFingerprints = dataProcessor.getOriginalTrainingApFingerprints()
    originalTrainingCoordinatesId = dataProcessor.getOriginalTrainingCoordinatesId()

    testingApFingerprints = dataProcessor.getTestingApFingerprints()
    testingCoordinatesId = dataProcessor.getTestingCoordinatesId()

    originalTestingApFingerprints = dataProcessor.getOriginalTestingApFingerprints()
    originalTestingCoordinatesId = dataProcessor.getOriginalTestingCoordinatesId()


    np.save('./Data/Original/trainingApFingerprints',trainingApFingerprints)
    np.save('./Data/Original/trainingCoordinatesId',trainingCoordinatesId)
    np.save('./Data/Original/originalTrainingApFingerprints',originalTrainingApFingerprints)
    np.save('./Data/Original/originalTrainingCoordinatesId',originalTrainingCoordinatesId)
    np.save('./Data/Original/testingApFingerprints',testingApFingerprints)
    np.save('./Data/Original/testingCoordinatesId',testingCoordinatesId)
    np.save('./Data/Original/originalTestingApFingerprints',originalTestingApFingerprints)
    np.save('./Data/Original/originalTestingCoordinatesId',originalTestingCoordinatesId)
    print('保存完毕！')

    '''
    #载入数据
    trainingApFingerprints = np.load('./Data/trainingApFingerprints.npy')
    trainingCoordinatesId = np.load('./Data/trainingCoordinatesId.npy') #one-hot vector
    trainingCoordinatesId = np.argmax(trainingCoordinatesId,axis=1)
    testingApFingerprints = np.load('./Data/testingApFingerprints.npy')
    testingCoordinatesId = np.load('./Data/testingCoordinatesId.npy') #one-hot vector
    testingCoordinatesId = np.argmax(testingCoordinatesId,axis=1)




    from sklearn.decomposition import PCA
    from sklearn.decomposition import KernelPCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.cluster import KMeans

    #PCA降维
    #pca = PCA(n_components=35)
    pca = PCA()
    pca_tsfm_trainingApFingerprints = pca.fit(trainingApFingerprints).transform(trainingApFingerprints)
    pca_tsfm_testingApFingerprints = pca.transform(testingApFingerprints)
    #print ('PCA explained variance ratio: %s' % str(pca.explained_variance_ratio_))
    #cum_ratio = np.cumsum(pca.explained_variance_ratio_)/np.sum(pca.explained_variance_ratio_)
    #print (np.where(cum_ratio>=0.95))

    #LDA降维
    lda = LDA()#n_components=52)#,solver='eigen',shrinkage=0.1)
    lda_tsfm_trainingApFingerprints = lda.fit(trainingApFingerprints,trainingCoordinatesId).transform(trainingApFingerprints)
    lda_tsfm_testingApFingerprints = lda.transform(testingApFingerprints)
    print ('LDA explained variance ratio: %s' % str(lda.explained_variance_ratio_))
    cum_ratio = np.cumsum(lda.explained_variance_ratio_)/np.sum(lda.explained_variance_ratio_)
    print(np.where(cum_ratio>=0.9))

    #不使用降维进行Kmeans
    #kmeans = KMeans(n_clusters=15).fit(trainingApFingerprints)
    #predict_labels = kmeans.predict(testingApFingerprints)
    #trainingCoordinatesId = dataToOne_hotVector(kmeans.labels_,trainingApFingerprints.shape[0],15)
    #testingCoordinatesId = dataToOne_hotVector(predict_labels,testingApFingerprints.shape[0],15)
    #print ('**********不使用降维进行KMeans')

    #使用PCA降维，再进行Kmeans
    pca_kmeans = KMeans(n_clusters=15).fit(pca_tsfm_trainingApFingerprints)
    pca_predict_labels = pca_kmeans.predict(pca_tsfm_testingApFingerprints)
    trainingCoordinatesId = dataToOne_hotVector(pca_kmeans.labels_,trainingApFingerprints.shape[0],15)
    testingCoordinatesId = dataToOne_hotVector(pca_predict_labels,testingApFingerprints.shape[0],15)
    print ('**********先使用PCA降维，再进行KMeans')

    #使用LDA降维，再进行Kmeans
    #lda_kmeans = KMeans(n_clusters=15).fit(lda_tsfm_trainingApFingerprints)
    #lda_predict_labels = lda_kmeans.predict(lda_tsfm_testingApFingerprints)
    #trainingCoordinatesId = dataToOne_hotVector(lda_kmeans.labels_,trainingApFingerprints.shape[0],15)
    #testingCoordinatesId = dataToOne_hotVector(lda_predict_labels,testingApFingerprints.shape[0],15)
    #print ('**********先使用LDA降维，再进行KMeans')





    #nn = bpnnPredictRpProbability(trainingApFingerprints,trainingCoordinatesId,testingApFingerprints,testingCoordinatesId)
    #设置网络结构
    #nn = setup(trainingApFingerprints.shape[1],[400],trainingCoordinatesId.shape[1])

    #LDA处理后的数据
    #from sklearn.preprocessing import StandardScaler
    #scaler = StandardScaler()
    #std_lda_tsfm_trainingApFingerprints = scaler.fit_transform(lda_tsfm_trainingApFingerprints)
    #print('************************标准化')
    #nn = bpnnPredictRpProbability(lda_tsfm_trainingApFingerprints,trainingCoordinatesId,lda_tsfm_testingApFingerprints,testingCoordinatesId)
    #设置网络的结构_LDA
    #nn.setup(lda_tsfm_trainingApFingerprints.shape[1],[400],trainingCoordinatesId.shape[1])

    #PCA处理后的数据
    nn = bpnnPredictRpProbability(pca_tsfm_trainingApFingerprints,trainingCoordinatesId,pca_tsfm_testingApFingerprints,testingCoordinatesId)
    #设置网络结构_PCA
    nn.setup(pca_tsfm_trainingApFingerprints.shape[1],[400],trainingCoordinatesId.shape[1])

    nn.trainAndTest()
    '''
