#!/usr/bin/env python
# coding=utf-8

from __future__ import division
import numpy as np
import tensorflow as tf
import time
import data_helper
import sys
import json

L1_loss = 0.0 #L1正则项
L2_loss = 0.0 #L2正则项

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
        #L1_loss += tf.reduce_sum(tf.abs(weights))
        #L2_loss += tf.nn.l2_loss(weights)
        #L1_loss += tf.reduce_sum(tf.abs(biases))
        #L2_loss += tf.nn.l2_loss(biases)
        L1_loss += tf.contrib.layers.l1_regularizer(1.0)(weights)
        L1_loss += tf.contrib.layers.l1_regularizer(1.0)(biases)
        L2_loss += tf.contrib.layers.l2_regularizer(1.0)(weights)
        L2_loss += tf.contrib.layers.l2_regularizer(1.0)(biases)

    return outputs

def nn_train(train_set,train_label,test_set,test_label):

    '''train step 0: load train_set,train_label,test_set,test_label, global coord list, local rp id set of this cluster
                     and parameters.
    '''
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
        loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(label_-output_layer),axis=1))) + params['lambda_l1'] * L1_loss + params['lambda_l2'] * L2_loss #针对只有一个正确答案的分类更高效

    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(params['learn_rate']).minimize(loss)

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
                #summary_writer.add_summary(summary,epoch)

            #进行测试
            train_loss_ = sess.run(loss,feed_dict = {input_:train_set,label_:train_label,keep_prob:1.0})
            test_loss_ = sess.run(loss,feed_dict={input_:test_set,label_:test_label,keep_prob:1.0})
            print ('epoch', epoch+1, 'train mean dist error:',train_loss_)
            print ('epoch', epoch+1, 'test mean dist error:', test_loss_)
            print ('*'*30)
            print ('')

        end_time = time.time()
        print ('*' *60)
        print ('Training finish! Cost time:', int(end_time-start_time) , 'seconds')
        print ('Training mean dist error:', sess.run(loss,feed_dict = {input_:train_set,label_:train_label,keep_prob:1.0}))
        print ('Testing mean dist error:', sess.run(loss,feed_dict={input_:test_set,label_:test_label,keep_prob:1.0}))
        print ('input dimension:',train_set.shape[1])
        print ('hidden dimension:',hidden_dims)
        print ('output dimension:',train_label.shape[1])
        print ('learn_rate:',params['learn_rate'])
        print ('keep_prob:',params['keep_prob'])
        print ('lambda_l1:',params['lambda_l1'])
        print ('lambda_l2:',params['lambda_l2'])
        print ('batch_size:',params['batch_size'])
        print ('cumulative probability threshold:',params['threshold'])
        print ('real coordinates vs estmated coordinates:')
        est_coords = sess.run(output_layer,feed_dict={input_:test_set,label_:test_label,keep_prob:1.0})
        for i in range(test_label.shape[0]):
            print(np.round(test_label[i],2),'  ',np.round(est_coords[i],2))


def main(_):
    '''step 0: load train_set,train_label and test_set, test_label for each cluster'''
    '''step 0.1:load PCA-transformed data'''
    train_fgprts_1 = np.load('./Data_Statistics/Fgprt_Rp4Cluster/PCA/train_fingerprints_1.npy')  #train set for cluster, after PCA reduction
    train_rp_ids_1 = np.load('./Data_Statistics/Fgprt_Rp4Cluster/PCA/train_rp_ids_1.npy')        #rp coordinate id in train set
    test_fgprts_1 = np.load('./Data_Statistics/Fgprt_Rp4Cluster/PCA/test_fingerprints_1.npy')    #test set for cluster, after PCA reduction
    test_rp_ids_1 = np.load('./Data_Statistics/Fgprt_Rp4Cluster/PCA/test_rp_ids_1.npy')          #rp coordinate if in test set
    coord_list = np.load('./Data/Original/rpCoordinatesList.npy')                                #global coordinatesList for all fingerprints
    train_rp_coords_1 = coord_list[train_rp_ids_1]
    test_rp_coords_1 = coord_list[test_rp_ids_1]

    '''step 0.2:load LDA-transformed data'''
    #train_fgprts_1 = np.load('./Data_Statistics/Fgprt_Rp4Cluster/LDA/train_fingerprints_1.npy')  #train set for cluster, after LDA reduction
    #train_rp_ids_1 = np.load('./Data_Statistics/Fgprt_Rp4Cluster/LDA/train_rp_ids_1.npy')        #rp coordinate id in train set
    #test_fgprts_1 = np.load('./Data_Statistics/Fgprt_Rp4Cluster/LDA/test_fingerprints_1.npy')    #test set for cluster, after LDA reduction
    #test_rp_ids_1 = np.load('./Data_Statistics/Fgprt_Rp4Cluster/LDA/test_rp_ids_1.npy')          #rp coordinate if in test set
    #coord_list = np.load('./Data/Original/rpCoordinatesList.npy')                               #global coordinatesList for all fingerprints
    #local_rp_id_set = np.unique(train_rp_ids_1)                                                 #local coordinateList for this cluster

    '''step 1: transform the train_rp_ids to one-hot vectors for nn training'''
    #train_label = train_data_to_one_hot_vector(train_rp_ids_1,local_rp_id_set)

    '''step 2: train neural network and test'''
    nn_train(train_fgprts_1,train_rp_coords_1,test_fgprts_1,test_rp_coords_1)


if __name__ == '__main__':
    #python3 second_staged_nn.py ./second_parameters.json
    tf.app.run()
