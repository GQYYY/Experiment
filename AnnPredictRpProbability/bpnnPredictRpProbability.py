#!/usr/bin/env python
# coding=utf-8

import numpy as np
import tensorflow as tf
import dataProcessor as dp
import time
import data_helper

L1_loss = 0.0 #L1正则项
L2_loss = 0.0 #L2正则项

def make_layer(inputs, input_size, output_size, activate=None,keep_prob=1.0):

		weights = tf.Variable(tf.random_uniform([input_size, output_size],-1.0,1.0))

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


		self.hidden_layers.append(make_layer(inputs, in_size, out_size, activate=tf.nn.relu,keep_prob=self.keep_prob))
		for i in range(self.hidden_n-1):
			in_size = out_size
			out_size = self.hidden_size[i+1]
			inputs = self.hidden_layers[-1]
			self.hidden_layers.append(make_layer(inputs, in_size, out_size, activate=tf.nn.relu,keep_prob=self.keep_prob))

		# build output layer
		self.output_layer = make_layer(self.hidden_layers[-1], self.hidden_size[-1], self.output_n,activate=tf.nn.softmax)


	def trainAndTest(self,epochs=100,learn_rate=1e-4,keep_prob=0.8,batch_size=64,lambda_l1=0.0,lambda_l2=0.1):

		print 'train_inputs.shape:',self.train_inputs.shape
		print 'train_outputs.shape:',self.train_outputs.shape
		print 'test_inputs.shape:',self.test_inputs.shape
		print 'test_outputs.shape:',self.test_outputs.shape

		#使用cross_entropy函数作为loss函数
		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.output_layer, self.label_layer)) + lambda_l1 * L1_loss +lambda_l2 * L2_loss   #
		#self.loss = tf.reduce_mean(-tf.reduce_sum(self.label_layer * tf.log(self.output_layer))) + lambda_l1 * L1_loss +lambda_l2 * L2_loss                  #

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

			print ''
			print '*'*30
			#print 'output:'
			#print self.session.run(self.output_layer,feed_dict={self.input_layer:self.train_inputs,self.label_layer:self.train_outputs,self.keep_prob:1.0})
			print 'output argmax:'
			print self.session.run(tf.argmax(self.output_layer,1),feed_dict = {self.input_layer:self.train_inputs,self.label_layer:self.train_outputs,self.keep_prob:1.0})[:50]
			print 'label argmax:'
			print self.session.run(tf.argmax(self.label_layer,1),feed_dict={self.label_layer:self.train_outputs})[:50]
			
			#进行测试
			print 'epoch', epoch+1, 'train accuracy:', self.session.run(accuracy, feed_dict={self.input_layer: self.train_inputs, self.label_layer: self.train_outputs,self.keep_prob:1.0})
			print 'epoch', epoch+1, 'test accuracy:', self.session.run(accuracy,feed_dict={self.input_layer: self.test_inputs, self.label_layer:self.test_outputs,self.keep_prob:1.0})
			print '*'*30
			print ''

		end_time = time.time()
		print '*' *60
		print 'Training finish! Cost time:', int(end_time-start_time) , 'seconds'
		print 'Training accuracy:', self.session.run(accuracy, feed_dict={self.input_layer: self.train_inputs, self.label_layer: self.train_outputs,self.keep_prob:1.0})		
		print 'Testing accuracy:', self.session.run(accuracy, feed_dict={self.input_layer: self.test_inputs, self.label_layer: self.test_outputs,self.keep_prob:1.0})
		print 'input dimension:',self.input_n
		print 'hidden dimension:',self.hidden_size
		print 'output dimension:',self.output_n
		print 'learn_rate:',learn_rate
		print 'keep_prob:',keep_prob
		print 'lambda_l1:',lambda_l1
		print 'lambda_l2:',lambda_l2
		print 'batch_size:',batch_size

def dataToOne_hotVector(_label,row,col):
	print row,col
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
	#traingingCoordinates = dataProcessor.getTrainingCoordinates()
	#获取和训练指纹所对应的RP的编号(12400*124)
	trainingCoordinatesId = dataProcessor.getTrainingCoordinatesId()
	#获取测试指纹
	testingApFingerprints = dataProcessor.getTestingApFingerprints() + 100
	#获取和训练指纹所对应的RP的编号
	testingCoordinatesId = dataProcessor.getTestingCoordinatesId()
	#testingCoordinates = dataProcessor.getTestingCoordinates()

	np.save('./Data/trainingApFingerprints',trainingApFingerprints)
	np.save('./Data/trainingCoordinatesId',trainingCoordinatesId)
	np.save('./Data/traingingCoordinates',traingingCoordinates)
	np.save('./Data/testingApFingerprints',testingApFingerprints)
	np.save('./Data/testingCoordinatesId',testingCoordinatesId)
	np.save('./Data/testingCoordinates',testingCoordinates)
	print '保存完毕'
	'''
	#载入数据
	trainingApFingerprints = np.load('./Data/trainingApFingerprints.npy')
	trainingCoordinatesId = np.load('./Data/trainingCoordinatesId.npy') #one-hot vector
	trainingCoordinatesId = np.argmax(trainingCoordinatesId,axis=1)
	testingApFingerprints = np.load('./Data/testingApFingerprints.npy')
	testingCoordinatesId = np.load('./Data/testingCoordinatesId.npy') #one-hot vector 
	testingCoordinatesId = np.argmax(testingCoordinatesId,axis=1)	

	'''
	#***********************tensorflow_KMeans by LHT
	from tf_kmeans import TFKMeans as KMeans
	kms = KMeans(k=15)
	#kms.train(trainingApFingerprints)
	kms.train(traingingCoordinates)
	np.save('tf_centers.npy', kms._centers)
	np.save('tf_labels.npy', kms._label)

	_label = np.load("tf_labels.npy")
	print _label[:200]
	trainingCoordinatesId = dataToOne_hotVector(_label, trainingApFingerprints.shape[0], 15)
	print trainingCoordinatesId
	'''


	from sklearn.decomposition import PCA
	from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
	from sklearn.cluster import KMeans

	#PCA降维
	pca = PCA()
	pca_tsfm_trainingApFingerprints = pca.fit(trainingApFingerprints).transform(trainingApFingerprints)
	pca_tsfm_testingApFingerprints = pca.fit(testingApFingerprints).transform(testingApFingerprints)
	#print 'PCA explained variance ratio: %s' % str(pca.explained_variance_ratio_)

	#LDA降维
	lda = LDA()
	lda_tsfm_trainingApFingerprints = lda.fit(trainingApFingerprints,trainingCoordinatesId).transform(trainingApFingerprints)
	lda_tsfm_testingApFingerprints = lda.fit(testingApFingerprints,testingCoordinatesId).transform(testingApFingerprints)
	#print 'LDA explained variance ratio: %s' % str(lda.explained_variance_ratio_)

	#不使用降维进行Kmeans
	# kmeans = KMeans(n_clusters=15).fit(trainingApFingerprints)
	# predict_labels = kmeans.predict(testingApFingerprints)
	# trainingCoordinatesId = dataToOne_hotVector(kmeans.labels_,trainingApFingerprints.shape[0],15)
	# testingCoordinatesId = dataToOne_hotVector(predict_labels,testingApFingerprints.shape[0],15)
	# print '**********不使用降维进行KMeans'

	#使用PCA降维，再进行Kmeans
	pca_kmeans = KMeans(n_clusters=15).fit(pca_tsfm_trainingApFingerprints)
	pca_predict_labels = pca_kmeans.predict(pca_tsfm_testingApFingerprints)
	trainingCoordinatesId = dataToOne_hotVector(pca_kmeans.labels_,trainingApFingerprints.shape[0],15)
	testingCoordinatesId = dataToOne_hotVector(pca_predict_labels,testingApFingerprints.shape[0],15)
	print '**********先使用PCA降维，再进行KMeans'

	#使用LDA降维，再进行Kmeans
	# lda_kmeans = KMeans(n_clusters=15).fit(lda_tsfm_traingApFingerprints)
	# lda_predict_labels = lda_kmeans.predict(lda_tsfm_testingApFingerprints)
	# trainingCoordinatesId = dataToOne_hotVector(lda_kmeans.labels_,trainingApFingerprints.shape[0],15)
	# testingCoordinatesId = dataToOne_hotVector(lda_predict_labels,testingApFingerprints.shape[0],15)
	# print '**********先使用LDA降维，再进行KMeans'

	

	
	nn = bpnnPredictRpProbability(trainingApFingerprints,trainingCoordinatesId,testingApFingerprints,testingCoordinatesId)
	#设置网络的结构
	nn.setup(trainingApFingerprints.shape[1],[400],trainingCoordinatesId.shape[1])
	nn.trainAndTest()