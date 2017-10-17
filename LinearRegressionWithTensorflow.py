# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 23:13:17 2017

@author: Sandesh
"""

import tensorflow as tf
import numpy as np

x = tf.placeholder(dtype=tf.float32, shape=[None, 13])
y = tf.placeholder(dtype=tf.float32, shape=[None, 1])


W = tf.Variable(tf.random_normal([13, 1]))
b = tf.Variable(tf.random_normal([1]))
y_pred = tf.matmul(x, W) + b
loss = tf.reduce_mean(tf.square(y - y_pred))
opt = tf.train.GradientDescentOptimizer(learning_rate = .5).minimize(loss)

from sklearn.datasets import load_boston
from sklearn.preprocessing import normalize # to standardize our data
from sklearn.model_selection import train_test_split
data, targets = load_boston(True)
data = normalize(data)
targets = targets.reshape((targets.shape[0],1)) # reshape targets to follow our variables
X_train, X_test, y_train, y_test = train_test_split(data, targets, 
                                                    test_size = 0.3, random_state = 42)
                                                    
numEpochs = 5000
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for i in range(numEpochs):
    sess.run(opt, feed_dict={x: X_train, y: y_train})
    if (i % 250 == 0):
        print ('Loss:', loss.eval(feed_dict={x: X_train, y: y_train}))


predictions = sess.run(y_pred, feed_dict={x: X_test})
differences = predictions.flatten() - y_test.flatten()
differences = [abs(x) for x in differences]
print ("House prices are off by an average of", np.mean(differences), "thousand dollars.")
                                                    