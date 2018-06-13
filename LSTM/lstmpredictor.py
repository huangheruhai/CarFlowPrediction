#!/usr/bin/env python
#-*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

myloss=[]
rnn_unit=128
input_size=4
output_size=1
lr=0.006

##----------load data-----------------
f=np.genfromtxt("D:\\on_line_car.txt",delimiter=',')
data=f[:,1:6]#get the second - fifth column

def get_train_data(batch_size=5,time_step=10,train_begin=0,train_end=320):
    batch_index=[]
    data_train=data[train_begin:train_end]
    normalized_train_data=(data_train-np.mean(data_train ,axis=0))/np.std(data_train,axis=0)
    train_x,train_y=[],[]
    for i in range(len(normalized_train_data)-time_step):
        if i % batch_size ==0:
            batch_index.append(i)
        x=normalized_train_data[i:i+time_step,1:5]
        y=normalized_train_data[i:i+time_step,0,np.newaxis]

        train_x.append(x.tolist())
        train_y.append(y.tolist())
    batch_index.append(len(normalized_train_data)-time_step)
    return batch_index,train_x,train_y

def get_test_data(time_step=10,test_begin=345):
    data_test=data[test_begin:]
    mean=np.mean(data_test,axis=0)
    std=np.std(data_test,axis=0)
    normalized_test_data=(data_test-np.mean(data_test ,axis=0))/np.std(data_test,axis=0)
    size=(len(normalized_test_data)+time_step-1)//time_step
    test_x,test_y=[],[]
    for i in range(size-1):
        x=normalized_test_data[i*time_step:(i+1)*time_step,1:5]
        y=normalized_test_data[i*time_step:(i+1)*time_step,0]
        test_x.append(x.tolist())
        test_y.append(y.tolist())

    return mean,std,test_x,test_y

##-------------neural-network--weight initialize---------------
weights={
    'in':tf.get_variable('in',shape=[input_size,rnn_unit],initializer=tf.orthogonal_initializer),
    'hidden':tf.get_variable('hidden',shape=[rnn_unit,rnn_unit],initializer=tf.orthogonal_initializer),
    'out':tf.get_variable('out',shape=[rnn_unit,1],initializer=tf.orthogonal_initializer)
}

biases={
    'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
    'out':tf.Variable(tf.constant(0.1,shape=[1,]))
}

##------------------network--structure------
def lstm(X):
    batch_size = tf.shape(X)[0]
    time_step = tf.shape(X)[1]
    w_in = weights['in']
    b_in = biases['in']
    input = tf.reshape(X, [-1, input_size])  # change tensor to 2 dim
    input_rnn = tf.matmul(input, w_in) + b_in
    input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])
    cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    drop=tf.nn.rnn_cell.DropoutWrapper(cell,output_keep_prob=0.5)
    cell=tf.nn.rnn_cell.MultiRNNCell([drop]*5,state_is_tuple=True)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state,
                                                 dtype=tf.float32)
    output = tf.reshape(output_rnn, [-1, rnn_unit])
    w_out = weights['out']
    b_out = biases['out']
    pred = tf.matmul(output, w_out) + b_out
    return pred, final_states

##---------------model trainning-----------------
def train_lstm(batch_size=80,time_step=60,train_begin=0,train_end=345):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
    batch_index,train_x,train_y=get_train_data(batch_size,time_step,train_begin,train_end)
    with tf.variable_scope('my_net_lstm'):

        pred,_=lstm(X)

    loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    saver=tf.train.Saver(tf.global_variables(),max_to_keep=15)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(10000):
            for step in range(len(batch_index)-1):
                _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],Y:train_y[batch_index[step]:batch_index[step+1]]})
            # print(i,loss_)
            myloss.append(loss_)
        plt.figure()
        plt.plot(myloss,color='b')
        plt.show()
        print("model_save: ：",saver.save(sess,'D:\\mymodel.ckpt'))
train_lstm()


##------------------prediction-------------

def prediction(time_step=5):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    mean,std,test_x,test_y=get_test_data(time_step)
    with tf.variable_scope('my_net_lstm',reuse=True):
        pred,_=lstm(X)
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        module_file = tf.train.latest_checkpoint('D:\\mymodel.ckpt')
        saver.restore(sess, module_file)
        test_predict=[]
        for step in range(len(test_x)-1):
          prob=sess.run(pred,feed_dict={X:[test_x[step]]})
          predict=prob.reshape((-1))
          test_predict.extend(predict)
        test_y=np.array(test_y)*std[0]+mean[0]
        test_predict=np.array(test_predict)*std[0]+mean[0]
        acc=np.average(np.abs(test_predict-test_y[:len(test_predict)])/test_y[:len(test_predict)])
        print('The accuracy of this predictor is :',acc)