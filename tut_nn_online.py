#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 11:40:09 2018

@author: aradulescu
"""

import tensorflow as tf

# Initialize two constants
x1 = tf.constant([1,2,3,4])
x2 = tf.constant([5,6,7,8])

# Multiply
result = tf.multiply(x1, x2)

# Print the result
print(result)

## Intialize the Session
#sess = tf.Session()
#
## Print the result
#print(sess.run(result))
#
## Close the session
#sess.close()

# Initialize Session and run `result`
with tf.Session() as sess:
  output = sess.run(result)
  print(output)