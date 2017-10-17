# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 01:21:15 2017

@author: Sandesh
"""

import tensorflow as tf

# Import MINST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

numClasses = 10 # The number of categories the model is choosing from
inputSize = 784 # A 28x28 image will have 784 total pixel values
numHiddenUnits = 50 # Number of hidden units this one layer NN will have
trainingIterations = 10000 # Number of times the training loop is run
batchSize = 100 # Represents how many images we feed in one training batch

tf.reset_default_graph() # Just to make sure that we start with a new graph
X = tf.placeholder(tf.float32, shape = [None, inputSize])
y = tf.placeholder(tf.float32, shape = [None, numClasses])


W1 = tf.Variable(tf.truncated_normal([inputSize, numHiddenUnits], stddev=0.1))
B1 = tf.Variable(tf.constant(0.1), [numHiddenUnits])
W2 = tf.Variable(tf.truncated_normal([numHiddenUnits, numClasses], stddev=0.1))
B2 = tf.Variable(tf.constant(0.1), [numClasses])

hiddenLayerOutput = tf.matmul(X, W1) + B1
hiddenLayerOutput = tf.nn.relu(hiddenLayerOutput)
finalOutput = tf.matmul(hiddenLayerOutput, W2) + B2
finalOutput = tf.nn.relu(finalOutput)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = finalOutput))
opt = tf.train.GradientDescentOptimizer(learning_rate = .1).minimize(loss)

correct_prediction = tf.equal(tf.argmax(finalOutput,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(trainingIterations):
    batch = mnist.train.next_batch(batchSize)
    batchInput = batch[0]
    batchLabels = batch[1]
    _, trainingLoss = sess.run([opt, loss], feed_dict={X: batchInput, y: batchLabels})
    if i%1000 == 0:
        trainAccuracy = accuracy.eval(session=sess, feed_dict={X: batchInput, y: batchLabels})
        print ("step %d, training accuracy %g"%(i, trainAccuracy))


testInputs = mnist.test.images
testLabels = mnist.test.labels
acc = accuracy.eval(session=sess, feed_dict = {X: testInputs, y: testLabels})
print("testing accuracy: {}".format(acc))