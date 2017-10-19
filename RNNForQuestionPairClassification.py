# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 13:26:10 2017

@author: Sandesh
"""

#Analyse the structure of the data

import pandas as pd
df = pd.read_csv("quora_duplicate_questions.tsv",sep="\t")

df.head()

df[df["is_duplicate"]==1].head()
df.shape

#Find number of duplicate and non duplicate questions
print("Number of duplicate pairs:", len(df[df["is_duplicate"]==1]))
print("Number of non duplicate pairs:", len(df[df["is_duplicate"]==0]))


import numpy as np
wordsList = np.load('Data/Quora/wordsList.npy').tolist()

#Get the pre-trained word vectors
for i in range(len(wordsList)):
    wordsList[i] = wordsList[i].decode('UTF-8')
wordVectors = np.load('Data/Quora/wordVectors.npy')
print(len(wordsList)) # Contains all of the words that we have vectors for
print(wordVectors.shape) # Contains all of the respective vectors
numDimensions = wordVectors.shape[1]
baseballIndex = wordsList.index('baseball')

firstQuestion = df.loc[7,'question1'] # Getting the first sentence in the first question pair
print ('The first question:', firstQuestion)
secondQuestion = df.loc[7,'question2'] # Getting the second sentence in the first question pair
print ('The second question:', secondQuestion)

#Clean up the sentences. Remove punctuation and upper case letters
import re
def cleanSentences(string):
    if(isinstance(string,str) == False):
        return " "
    string = string.lower()
    string = re.sub('([.,!?()])', r' \1 ', string) # Separates punctuation from the word
    return string

firstQuestion = cleanSentences(firstQuestion)
secondQuestion = cleanSentences(secondQuestion)
firstQuestionSplit = firstQuestion.split()
secondQuestionSplit = secondQuestion.split()
lenBothSentences = len(firstQuestionSplit) + len(secondQuestionSplit)
print("The total number of words:", lenBothSentences)

#Create matrices of word vectors for the first instance
firstXInput = np.zeros((lenBothSentences, numDimensions), dtype = 'float32')
indexCounter = 0
for word in firstQuestionSplit:
    try:
        firstXInput[indexCounter] = wordVectors[wordsList.index(word)]
    except ValueError:
        firstXInput[indexCounter] = wordVectors[399999]
    indexCounter = indexCounter+1

for word in secondQuestionSplit:
    try:
        firstXInput[indexCounter] = wordVectors[wordsList.index(word)]
    except:
        firstXInput[indexCounter] = wordVectors[399999]
    indexCounter = indexCounter+1

firstXInput.shape

#Find maximum sequence length among all sentences in the dataframe.
maxSeqLength = 0
for index, row in df.iterrows():
    firstQuestion = cleanSentences(firstQuestion)
    secondQuestion = cleanSentences(secondQuestion)
    firstQuestionSplit = firstQuestion.split()
    secondQuestionSplit = secondQuestion.split()
    lenBothSentences = len(firstQuestionSplit) + len(secondQuestionSplit)
    if(lenBothSentences > maxSeqLength):
        maxSeqLength = lenBothSentences
print ('The maximum sequence length in the whole dataset is:', maxSeqLength)

maxSeqLength = 296 #Workaround
numTrainExamples = 350000
numTestExamples = df.shape[0] - numTrainExamples

#Create X and Y matrices for training
numClasses = 2
X = np.zeros((numTrainExamples + numTestExamples, maxSeqLength), dtype='int64')
Y = np.zeros((numTrainExamples + numTestExamples, numClasses), dtype='int32')

#Fill in the matrices from the dataframe
exampleCounter = 0
for index,row in df.iterrows():
    firstQuestion = cleanSentences(row['question1'])
    secondQuestion = cleanSentences(row['question2'])
    firstQuestionSplit = firstQuestion.split()
    secondQuestionSplit = secondQuestion.split()
    indexCounter = 0
    for word in firstQuestionSplit:
        try:
            X[exampleCounter][indexCounter] = wordsList.index(word)
        except ValueError:
            X[exampleCounter] [indexCounter] = 399999
        indexCounter = indexCounter+1
    
    for word in secondQuestionSplit:
        try:
            X[exampleCounter][indexCounter] = wordsList.index(word)
        except ValueError:
            X[exampleCounter][indexCounter] = 399999
        indexCounter = indexCounter + 1
    if(row['is_duplicate'] == 1):
        Y[exampleCounter] = [0,1]
    else:
        Y[exampleCounter] = [1,0]
    exampleCounter = exampleCounter + 1

np.save('Data/Quora/xMatrix.npy', X)
np.save('Data/Quora/yMatrix.npy', Y)

X = np.load('Data/Quora/xMatrix.npy')
Y = np.load('Data/Quora/yMatrix.npy')

#Split into training and test set
X_train = X[0:numTrainExamples]
Y_train = Y[0:numTrainExamples]
X_test = X[numTrainExamples:]
Y_test = Y[numTrainExamples:]

batchSize = 24
lstmUnits = 64
iterations = 10000
#Helper functions
from random import randint

def getBatch(isTest=False):
    labels = np.zeros([batchSize, numClasses])
    arr = np.zeros([batchSize, maxSeqLength])
    if(isTest):
        num = randint(0,X_test.shape[0] - batchSize)
        arr = X_test[num:num+batchSize]
        labels = Y_test[num:num+batchSize]
    else:
        num = randint(0,X_train.shape[0] - batchSize)
        arr = X_train[num:num+batchSize]
        labels = Y_train[num:num+batchSize]
    return arr, labels

import tensorflow as tf
tf.reset_default_graph()
labels = tf.placeholder(tf.float32, [batchSize,numClasses])
input_data = tf.placeholder(tf.int32, [batchSize,maxSeqLength])

data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),
                   dtype=tf.float32)
    
data = tf.nn.embedding_lookup(wordVectors,input_data)

lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

hiddenUnits = 32

weight = tf.Variable(tf.truncated_normal([lstmUnits,hiddenUnits]))
bias = tf.Variable(tf.constant(0.1,shape=[hiddenUnits]))
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
fc1 = (tf.matmul(last, weight) + bias)

weight2 = tf.Variable(tf.truncated_normal([hiddenUnits, numClasses]))
bias2 = tf.Variable(tf.constant(0.1, shape=[numClasses]))
prediction = (tf.matmul(fc1, weight2) + bias2)

correctPred = tf.equal(tf.arg_max(prediction,1), tf.arg_max(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred,tf.float32))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,
                                                              labels=labels))

optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

import datetime
sess = tf.InteractiveSession()
tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)

saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

for i in range(iterations):
    nextBatch, nextBatchLabels = getBatch()
    sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})
    if(i%10 == 0):
        summary = sess.run(merged, {input_data: nextBatch, labels:nextBatchLabels})
        writer.add_summary(summary,i)
    if (i % 1000 == 0):
        trainingAccuracy = sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})
        print ('The training loss at iteration', i, 'is', trainingAccuracy)
writer.close()

iterations = 10
for i in range(iterations):
    nextBatch, nextBatchLabels = getBatch(True);
    print("Accuracy for this batch:", (sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})) * 100)