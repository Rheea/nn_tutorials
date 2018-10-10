#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 13:38:55 2018

@author: aradulescu
"""
import tensorflow as tf
import numpy as np
NO_OF_RANDOM_POINTS = 100
CIRCLE_RADIUS = 0.5
random_spots = np.random.rand(NO_OF_RANDOM_POINTS, 2) * 2 - 1
is_inside_circle = (np.power(random_spots[:,0],2) + np.power(random_spots[:,1],2) < CIRCLE_RADIUS).astype(int)


INPUT_LAYER_SIZE = 2
HIDDEN_LAYER_SIZE = 3
OUTPUT_LAYER_SIZE = 2
 
# Starting values for weights and biases are drawn randomly and uniformly from  [-1, 1]
# For example W1 is a matrix of shape 2x3
W1 = tf.Variable(tf.random_uniform([INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE], -1, 1))
b1 = tf.Variable(tf.random_uniform([HIDDEN_LAYER_SIZE], -1, 1))
W2 = tf.Variable(tf.random_uniform([HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE], -1, 1))
b2 = tf.Variable(tf.random_uniform([OUTPUT_LAYER_SIZE], -1, 1))
# Specifying that the placeholder X can expect a matrix of 2 columns (but any number of rows)
# representing random spots
X = tf.placeholder(tf.float32, [None, INPUT_LAYER_SIZE])
# Placeholder Y can expect integers representing whether corresponding point is in the circle
# or not (no shape specified)
Y = tf.placeholder(tf.uint8)
# An op to convert to a one hot vector
onehot_output = tf.one_hot(Y, OUTPUT_LAYER_SIZE)

LEARNING_RATE = 0.01
# Op to perform matrix calculation X*W1 + b1
hidden_layer = tf.add(tf.matmul(X, W1), b1)
# Use sigmoid activation function on the outcome
activated_hidden_layer = tf.sigmoid(hidden_layer)
# Apply next weights and bias (W2, b2) to hidden layer and then apply softmax function
# to get our output layer (each vector adding up to 1)
output_layer = tf.nn.softmax(tf.add(tf.matmul(activated_hidden_layer, W2), b2))
# Calculate cross entropy for our loss function
loss = -tf.reduce_sum(onehot_output * tf.log(output_layer))
# Use gradient descent optimizer at specified learning rate to minimize value given by loss tensor
train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
#train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

EPOCH_COUNT = 100000
sess = tf.Session()
tf.global_variables_initializer().run(session=sess)
for i in range(EPOCH_COUNT):
    if i%100 == 0:
    	print('Loss after %d runs: %f' % (i, sess.run(loss, feed_dict={X: random_spots, Y: is_inside_circle})))
	sess.run(train_step, feed_dict={X: random_spots, Y: is_inside_circle})
print('Final loss after %d runs: %f' % (i, sess.run(loss, feed_dict={X: random_spots, Y: is_inside_circle})))


sess.run(output_layer, feed_dict={X: [[1, 1]]}) # Hopefully something close to [1, 0]
sess.run(output_layer, feed_dict={X: [[0, 0]]}) # Hopefully something close to [0, 1]