# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 09:15:16 2017

@author: Sandesh
"""


import tensorflow as tf
import random
import numpy as np
import datetime

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
tf.reset_default_graph() 
sess = tf.InteractiveSession()
x = tf.placeholder("float", shape = [None, 784]) 
y_ = tf.placeholder("float", shape = [None, 10]) 
x_image = tf.reshape(x, [-1,28,28,1])
W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(.1, shape = [32]))
h_conv1 = tf.nn.conv2d(input=x_image, filter=W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1
h_conv1 = tf.nn.relu(h_conv1)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def conv2d(x, W):
  return tf.nn.conv2d(input=x, filter=W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
  
#Second Conv and Pool Layers
W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(.1, shape = [64]))
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


#First Fully Connected Layer
W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(.1, shape = [1024]))
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#Dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#Second fully conncted layer
W_fc2 = tf.Variable(tf.truncated_normal([1024,10],stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1,shape=[10]))

#Softmax layer
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#Cross Entropy loss function

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))

#Mean Squared Error loss function
MSE = tf.reduce_sum(tf.square(y_-y_conv))
train_step = tf.train.AdamOptimizer().minimize(MSE)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
sess.run(tf.global_variables_initializer())

#Noisy labels
changeLabelProbability=0.5

tf.summary.scalar('MSE_Loss', MSE)
tf.summary.scalar('Accuracy', accuracy)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + "ChangeLabelProb:" + str(changeLabelProbability) + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)

#Training iterations
batchSize = 50

for i in range(2000):
    batch = mnist.train.next_batch(batchSize)
    if i%100 == 0:
        train_accuracy = accuracy.eval(session=sess, feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        print ("step %d, training accuracy %g"%(i, train_accuracy))
        
    # NOISY LABELS
    if (random.random() < changeLabelProbability):
        for i in range(batchSize):
            originalTrainingLabel = np.argmax(batch[1][i])
            validOtherChoices = list(range(0,originalTrainingLabel)) + list(range(originalTrainingLabel+1,10))
            newTrainingLabel = random.choice(validOtherChoices)
            batch[1][i] = np.zeros(10)
            batch[1][i][newTrainingLabel] = 1
    train_step.run(session=sess, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.2})
    
#Test iterations

for i in range(10):
    batch = mnist.test.next_batch(50)
    print (sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1}))