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

#transform _label to one-hot vector
def train_data_to_one_hot_vector(train_rp_ids,rp_id_set):
    row = train_rp_ids.size
    col = rp_id_set.size
    one_hot_vectors = np.zeros([row,col])
    for i,rp_id in enumerate(train_rp_ids.tolist()):
        one_hot_vectors[i,np.where(rp_id_set == rp_id)] = 1.0
    return one_hot_vectors

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

def nn_train(train_set,train_label,train_rp_ids,test_set,test_rp_ids,coord_list,local_rp_id_set):

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
        #tf.scalar_summary('accuracy',accuracy)  #for tensorflow < 0.12
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
                #summary_writer.add_summary(summary,epoch)

            print ('')
            print ('*'*30)
            print ('output argmax:')
            print (sess.run(tf.argmax(output_layer,1),feed_dict = {input_:train_set,label_:train_label,keep_prob:1.0})[:50])
            print ('label argmax:')
            print (sess.run(tf.argmax(label_,1),feed_dict={label_:train_label})[:50])

            #进行测试
            loss_,train_accuracy_,train_output = sess.run([cross_entropy,accuracy,tf.nn.softmax(output_layer)],\
                                                         feed_dict = {input_:train_set,label_:train_label,keep_prob:1.0})
            train_mean_dist_err = estimate_result(train_output,train_rp_ids,params['threshold'],coord_list,local_rp_id_set)[0]
            test_output = sess.run(tf.nn.softmax(output_layer),feed_dict={input_:test_set,keep_prob:1.0})
            test_mean_dist_err = estimate_result(test_output,test_rp_ids,params['threshold'],coord_list,local_rp_id_set)[0]

            print ('epoch', epoch+1, 'loss:', loss_)
            print ('epoch', epoch+1, 'train accuracy:', train_accuracy_)
            print ('epoch', epoch+1, 'train mean dist error:',train_mean_dist_err)
            print ('epoch', epoch+1, 'test mean dist error:', test_mean_dist_err)
            print ('*'*30)
            print ('')

        end_time = time.time()
        test_output = sess.run(tf.nn.softmax(output_layer),feed_dict={input_:test_set,keep_prob:1.0})
        test_mean_dist_err,est_coords,real_coords = estimate_result(test_output,test_rp_ids,params['threshold'],coord_list,local_rp_id_set)
        print ('*' *60)
        print ('Training finish! Cost time:', int(end_time-start_time) , 'seconds')
        print ('Training accuracy:',sess.run(accuracy, feed_dict = {input_:train_set,label_:train_label,keep_prob:1.0}))
        print ('Testing mean dist error:', test_mean_dist_err)
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
        for i in range(real_coords.shape[0]):
            print(np.round(real_coords[i],2),'  ',np.round(est_coords[i],2))

def estimate_result(original_probs,rp_coord_id,threshold,coord_list,local_rp_id_set):
    '''
    original_probs:神经网络输出的概率向量
    rp_coord_id:真实的位置坐标在coord_list中对应的索引向量
    threshold:累积概率的阈值
    coord_list:全局的所有RP的坐标向量
    local_rp_id_set:该cluster中出现的所有RP的坐标在coord_list中的索引的向量
    '''
    sorted_probs = np.sort(original_probs,axis=1)[:,::-1]
    sorted_probs_indices = np.argsort(original_probs,axis=1)[:,::-1]
    print(sorted_probs)
    print(sorted_probs.dtype)
    top_k = np.asarray([np.where(np.round(np.cumsum(sorted_probs[i]),3) >= threshold)[0][0] for i in range(sorted_probs.shape[0])])+1
    print('**************************')
    print(top_k[:50])
    rp_coord_in_use = []      #参与预测位置计算的rp坐标
    weights = []              #各坐标对应的权重
    for probs_,indices_,top_k_ in zip(sorted_probs,sorted_probs_indices,top_k):
        top_k_probs = probs_[:top_k_]
        top_k_indices = indices_[:top_k_]
        rp_coord_in_use.append(coord_list[local_rp_id_set[top_k_indices]])
        weights.append(top_k_probs/np.sum(top_k_probs))
    rp_coord_in_use = np.array(rp_coord_in_use)
    weights = np.array(weights)
    print('------------------------------weights')
    print (weights[:50])
    est_coord = []
    for rp_coords_, weights_ in zip(rp_coord_in_use,weights):
        rp_x = rp_coords_[:,0]
        rp_y = rp_coords_[:,1]
        est_x = np.sum(rp_x*weights_)
        est_y = np.sum(rp_y*weights_)
        est_coord.append([est_x,est_y])
    est_coord = np.array(est_coord)      #预测的位置坐标

    real_coord = coord_list[rp_coord_id] #真实的位置坐标
    mean_dist_err = np.mean(np.sqrt(np.sum(np.square(est_coord-real_coord),axis=1))) #平均误差

    return mean_dist_err,est_coord, real_coord



def main(_):
    '''step 0: load train_set,train_label and test_set, test_label for each cluster'''
    train_fgprts_1 = np.load('./Data_Statistics/Fgprt_Rp4Cluster/train_fingerprints_1.npy')  #train set for cluster, after PCA reduction
    train_rp_ids_1 = np.load('./Data_Statistics/Fgprt_Rp4Cluster/train_rp_ids_1.npy')        #rp coordinate id in train set
    test_fgprts_1 = np.load('./Data_Statistics/Fgprt_Rp4Cluster/test_fingerprints_1.npy')    #test set for cluster, after PCA reduction
    test_rp_ids_1 = np.load('./Data_Statistics/Fgprt_Rp4Cluster/test_rp_ids_1.npy')          #rp coordinate if in test set
    coord_list = np.load('./Data/coordinatesList.npy')                                       #global coordinatesList for all fingerprints
    local_rp_id_set = np.unique(train_rp_ids_1)                                              #local coordinateList for this cluster


    '''step 1: transform the train_rp_ids to one-hot vectors for nn training'''
    train_label = train_data_to_one_hot_vector(train_rp_ids_1,local_rp_id_set)

    '''step 2: train neural network and test'''
    nn_train(train_fgprts_1,train_label,train_rp_ids_1,test_fgprts_1,test_rp_ids_1,coord_list,local_rp_id_set)


if __name__ == '__main__':
    #python3 second_staged_nn.py ./second_parameters.json
    tf.app.run()
