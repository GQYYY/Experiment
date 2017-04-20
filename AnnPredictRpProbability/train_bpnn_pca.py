#!/usr/bin/env python
# coding=utf-8

import numpy as np
import tensorflow as tf
import time
import data_helper
import sys
import json

L1_loss = 0.0 #L1正则项
L2_loss = 0.0 #L2正则项

#transform _label to one-hot vector
def dataToOne_hotVector(_label,row,col):
    hotVectors = np.zeros([row,col])
    for i,label in enumerate(_label.tolist()):
        hotVectors[i,int(label)] = 1.0
    return hotVectors

def nn_layer(inputs, input_dim, output_dim, layer_n=None,activate=None,keep_prob=1.0,name='hidden_layer'):
    if layer_n is not None:
        layer_name = name+str(layer_n)
    else:
        layer_name = name
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = tf.get_variable(layer_name+'/weights',shape=[input_dim,output_dim],initializer=tf.contrib.layers.xavier_initializer())
            #tf.histogram_summary(layer_name+'/weights',weights)   #for tensorflow < 0.12
            tf.summary.histogram('weights',weights)   #for tensorflow >= 0.12
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.constant(0.1,shape=[1,output_dim]))
            #tf.histogram_summary(layer_name+'/biases',biases)  #for tensorflow < 0.12
            tf.summary.histogram('biases',biases)  #for tensorflow >= 0.12

        Wx_plus_b = tf.matmul(inputs,weights) + biases
        if activate is None:
            outputs = Wx_plus_b
        else:
            outputs = activate(Wx_plus_b)

        outputs = tf.nn.dropout(outputs,keep_prob)
        #tf.histogram_summary(layer_name+'/outputs',outputs) #for tensorflow < 0.12
        tf.summary.histogram('outputs',outputs) #for tensorflow >= 0.12

        global L1_loss
        global L2_loss
        L1_loss += tf.reduce_sum(tf.abs(weights))
        L2_loss += tf.nn.l2_loss(weights)
        L1_loss += tf.reduce_sum(tf.abs(biases))
        L2_loss += tf.nn.l2_loss(biases)
    return outputs

def nn_train(train_set,train_label,test_set,test_label):

    '''train step 0: load train_set,train_label,test_set,test_label and parameters '''
    parameter_file = sys.argv[1]
    params = json.loads(open(parameter_file).read())
    '''train step 1: construct/initialize nn structure '''
    with tf.name_scope('input'):
        input_ = tf.placeholder(tf.float32, [None,train_set.shape[1]])
        label_ = tf.placeholder(tf.float32, [None, train_label.shape[1]])
        keep_prob = tf.placeholder(tf.float32)

    # build hidden layers
    hidden_dims = params['hidden_dim'] #各隐层的单元个数
    hidden_layers = []
    input_dim = train_set.shape[1]
    output_dim = params['hidden_dim'][0]
    hidden_layers.append(nn_layer(input_, input_dim, output_dim, 1,activate=tf.nn.relu,keep_prob=keep_prob))
    for i in range(len(hidden_dims)-1):
        input_dim = output_dim
        output_dim = hidden_dims[i+1]
        inputs = hidden_layers[-1]
        hidden_layers.append(nn_layer(inputs, input_dim, output_dim, i+1,activate=tf.nn.relu,keep_prob=keep_prob))

    #tf.nn.sparse_softmax_cross_entropy_with_logits()，要求神经网络的输出层不经过softmax处理
    output_layer = nn_layer(hidden_layers[-1], hidden_dims[-1],train_label.shape[1], activate=None,name='output_layer')

    '''trian step 2: train and test nn '''
    with tf.name_scope('loss'):
        #使用cross_entropy函数作为loss函数
        #cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output_layer, label_)) + lambda_l1 * L1_loss +lambda_l2 * L2_loss   #高版本tensorflow
        #cross_entropy = tf.reduce_mean(-tf.reduce_sum(label_ * tf.log(output_layer))) + lambda_l1 * L1_loss +lambda_l2 * L2_loss                   #低版本tensorflow
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(output_layer,tf.argmax(label_,1))) \
                + params['lambda_l1'] * L1_loss + params['lambda_l2'] * L2_loss #针对只有一个正确答案的分类更高效
        #tf.scaler_summary('cross_entropy',cross_entropy)  #for tensorflow < 0.12
        tf.summary.scalar('cross_entropy',cross_entropy)  #for tensorflow >= 0.12

    with tf.name_scope('train'):
        #optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(cross_entropy)
        optimizer = tf.train.AdamOptimizer(params['learn_rate']).minimize(cross_entropy)
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(output_layer, 1), tf.argmax(label_, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        #tf.scalar_summary('accuracy',accuracy)  #for tensorflo w < 0.12
        tf.summary.scalar('accuracy',accuracy)  #for tensorflow >= 0.12

    #将代码中定义的所有日志生成操作都执行一次
    merged = tf.merge_all_summaries()
    with tf.Session() as sess:
        #初始化写日志的writer,并将当前Tensorflow计算图写入日志
        #summary_writer = tf.train.SummaryWriter('logs/',sess.graph)
        summary_writer = tf.summary.FileWriter('logs/',sess.graph)
        tf.initialize_all_variables().run()
        #initer = tf.initialize_all_variables()
        #sess.run(initer)

        start_time = time.time()
        for epoch in range(params['epochs']):
            batches = data_helper.batch_iter(train_set, train_label, params['batch_size'])
            for batch in batches:
                x,y = batch
                summary , _ = sess.run([merged,optimizer], feed_dict={input_: x, label_: y,keep_prob:params['keep_prob']})
                #将所有日志写入文件，TensorBoard即可拿到这次运行所对应的运行信息
                summary_writer.add_summary(summary,epoch)

            print ('')
            print ('*'*30)
            print ('output argmax:')
            print (sess.run(tf.argmax(output_layer,1),feed_dict = {input_:train_set,label_:train_label,keep_prob:1.0})[:50])
            print ('label argmax:')
            print (sess.run(tf.argmax(label_,1),feed_dict={label_:train_label})[:50])

            #进行测试
            print ('epoch', epoch+1, 'loss:', sess.run(cross_entropy, feed_dict = {input_:train_set,label_:train_label,keep_prob:1.0}))
            print ('epoch', epoch+1, 'train accuracy:', sess.run(accuracy, feed_dict = {input_:train_set,label_:train_label,keep_prob:1.0}))
            print ('epoch', epoch+1, 'test accuracy:', sess.run(accuracy,feed_dict = {input_:test_set,label_:test_label,keep_prob:1.0}))
            print ('*'*30)
            print ('')

        end_time = time.time()
        print ('*' *60)
        print ('Training finish! Cost time:', int(end_time-start_time) , 'seconds')
        print ('Training accuracy:',sess.run(accuracy, feed_dict = {input_:train_set,label_:train_label,keep_prob:1.0}))
        print ('Testing accuracy:', sess.run(accuracy, feed_dict = {input_:test_set,label_:test_label,keep_prob:1.0}))
        print ('input dimension:',train_set.shape[1])
        print ('hidden dimension:',hidden_dims)
        print ('output dimension:',train_label.shape[1])
        print ('learn_rate:',params['learn_rate'])
        print ('keep_prob:',params['keep_prob'])
        print ('lambda_l1:',params['lambda_l1'])
        print ('lambda_l2:',params['lambda_l2'])
        print ('batch_size:',params['batch_size'])

    summary_writer.close()



def main(_):
    '''step 0: load original train_set,train_label and original test_set, test_label'''
    trainingApFingerprints = np.load('./Data/trainingApFingerprints.npy')         #train set
    trainingCoordinatesId = np.load('./Data/trainingCoordinatesId.npy')           #train label,one-hot vector
    trainingCoordinatesId = np.argmax(trainingCoordinatesId,axis=1)               #train label, array
    testingApFingerprints = np.load('./Data/testingApFingerprints.npy')           #test set
    testingCoordinatesId = np.load('./Data/testingCoordinatesId.npy')             #test label,one-hot vector
    testingCoordinatesId = np.argmax(testingCoordinatesId,axis=1)                 #test label, array
    coordinatesList = np.load('./Data/coordinatesList.npy')

    '''step 1: PCA feature reduction and then KMeans'''
    '''step 1.1: PCA feature reduction'''
    from sklearn.decomposition import PCA
    pca = PCA() #PCA feature reduction
    pca_tsfm_trainingApFingerprints = pca.fit(trainingApFingerprints).transform(trainingApFingerprints)
    pca_tsfm_testingApFingerprints = pca.transform(testingApFingerprints)
    #print ('PCA explained variance ratio: %s' % str(pca.explained_variance_ratio_))
    #cum_ratio = np.cumsum(pca.explained_variance_ratio_)/np.sum(pca.explained_variance_ratio_)
    #print (np.where(cum_ratio>=0.95))
    '''step 1.2: KMeans'''
    from sklearn.cluster import KMeans
    pca_tsfm_train_test_fingerpringts = np.concatenate((pca_tsfm_trainingApFingerprints,pca_tsfm_testingApFingerprints),axis=0)
    pca_kmeans = KMeans(n_clusters=15).fit(pca_tsfm_train_test_fingerpringts) #KMeans
    #pca_kmeans = KMeans(n_clusters=15).fit(pca_tsfm_trainingApFingerprints) #KMeans
    #pca_predict_labels = pca_kmeans.predict(pca_tsfm_testingApFingerprints)
    train_cluster_labels = dataToOne_hotVector(pca_kmeans.labels_[:trainingApFingerprints.shape[0]],trainingApFingerprints.shape[0],15)
    test_cluster_labels = dataToOne_hotVector(pca_kmeans.labels_[trainingApFingerprints.shape[0]:],testingApFingerprints.shape[0],15)
    print ('**********先使用PCA降维，再进行KMeans')

    '''step 1.3: Analyse the cluster each RP belongs to, and the set of RPs each cluster consist of'''
    #data_helper.cluster_anls(pca_kmeans.labels_,pca_tsfm_trainingApFingerprints,trainingCoordinatesId,coordinatesList)

    '''step 1.4: Separate corresponding fingerprints and rp_id for each cluster label'''
    for cluster_label in np.unique(pca_kmeans.labels_):
        train_indices = np.where(pca_kmeans.labels_[:trainingApFingerprints.shape[0]] == cluster_label)
        train_fingerprints = pca_tsfm_trainingApFingerprints[train_indices]
        train_rp_ids = trainingCoordinatesId[train_indices]
        test_indices = np.where(pca_kmeans.labels_[trainingApFingerprints.shape[0]:] == cluster_label)
        test_fingerprints = pca_tsfm_testingApFingerprints[test_indices]
        test_rp_ids = testingCoordinatesId[test_indices]
        np.save('./Data_Statistics/Fgprt_Rp4Cluster/train_fingerprints_%d'%cluster_label,train_fingerprints)
        np.save('./Data_Statistics/Fgprt_Rp4Cluster/train_rp_ids_%d'%cluster_label,train_rp_ids)
        np.save('./Data_Statistics/Fgprt_Rp4Cluster/test_fingerprints_%d'%cluster_label,test_fingerprints)
        np.save('./Data_Statistics/Fgprt_Rp4Cluster/test_rp_ids_%d'%cluster_label,test_rp_ids)

    '''step 2: Train neural network amZd test'''
    #nn_train(pca_tsfm_trainingApFingerprints,train_cluster_labels,pca_tsfm_testingApFingerprints,test_cluster_labels)


if __name__ == '__main__':
    #python3 train_bpnn_pca.py ./parameters.json

    tf.app.run()
