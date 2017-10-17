# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 23:16:16 2017

@author: Sandesh
"""

import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

numClasses=10
inputSize=784
trainingIterations=20000
batchSize=64
tf.reset_default_graph()

X = tf.placeholder(tf.float32,shape=[None,inputSize])
y = tf.placeholder(tf.float32,shape=[None,numClasses])

W1 = tf.Variable(tf.random_normal([inputSize,numClasses],stddev=0.1))
B1 = tf.Variable(tf.constant(0.1), [numClasses])

y_pred = tf.nn.softmax(tf.matmul(X,W1)+B1)

loss = tf.reduce_mean(tf.square(y-y_pred))
opt = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(loss)

correct_prediction = tf.equal(tf.arg_max(y_pred,1), tf.arg_max(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(trainingIterations):
    batch = mnist.train.next_batch(batchSize)
    batchInput = batch[0]
    batchLabels = batch[1]
    _, trainingLoss = sess.run([opt,loss], feed_dict={X:batchInput,
                               y:batchLabels})
    if i%1000 == 0:
        train_accuracy = accuracy.eval(session=sess, feed_dict={X: batchInput, y: batchLabels})
        print ("step %d, training accuracy %g"%(i, train_accuracy))

batch = mnist.test.next_batch(batchSize)
testAccuracy = sess.run(accuracy, feed_dict={X: batch[0], y: batch[1]})
print ("test accuracy %g"%(testAccuracy))