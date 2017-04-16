#!/usr/bin/env python
# coding=utf-8

import numpy as np
import tensorflow as tf
import time
import data_helper
import sys
import json

# store original data 
class DataHolder:
    def __init__(self,global_train_set,global_train_label,global_test_set,global_test_label):
        self.global_train_set_ = global_train_set
        self.global_train_label_ = global_train_label
        self.global_test_set_ = global_test_set
        self.global_test_label_ = global_test_label


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
            print ('epoch', epoch+1, 'train mean dist error:',sess.run(mean_dist_error(output_layer),feed_dict={input_:train_set,label_:train_label,keep_prob:1.0}))
            print ('epoch', epoch+1, 'test mean dist error:', sess.run(mean_dist_error(output_layer),feed_dict = {input_:test_set,label_:test_label,keep_prob:1.0}))
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

#距离平均误差
def mean_dist_error(output_layer):
    '''
    output_layer:神经网络的输出层,尚未经过softmax处理
    rp_id:对应输入的真实的rp_id
    '''
    probabilies = tf.nn.softmax(output_layer)
    print(probabilies)

def main(_):
    '''step 0: load train_set,train_label and test_set, test_label for each cluster'''
    train_fgprts_1 = np.load('./Data_Statistics/Fgprt_Rp4Cluster/train_fingerprints_1.npy')  #train set for cluster, after PCA reduction  
    train_rp_ids_1 = np.load('./Data_Statistics/Fgprt_Rp4Cluster/train_rp_ids_1.npy')        #rp coordinate id in train set
    test_fgprts_1 = np.load('./Data_Statistics/Fgprt_Rp4Cluster/test_fingerprints_1.npy')    #test set for cluster, after PCA reduction
    test_rp_ids_1 = np.load('./Data_Statistics/Fgprt_Rp4Cluster/test_rp_ids_1.npy')          #rp coordinate if in test set
    coord_list = np.load('./Data/coordinatesList.npy')                                       #global coordinatesList for all fingerprints
    local_coord_list = np.load('./Data_Statistics/Rp_Cluster_Relation/cluster_1.npy')        #local coordinateList for this cluster

    '''step 1: transform the r'''
    '''step 2: train neural network and test'''
    #nn_train()


if __name__ == '__main__':
    #python3 train_bpnn_pca.py ./parameters.json

    tf.app.run()
