#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 14:55:42 2018

@author: aradulescu
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
message = tf.constant('Welcome to the exciting world of Deep Neural Networks!')
v_1 = tf.constant([1,2,3,4]) 
v_2 = tf.constant([2,1,5,3]) 
v_add = tf.add(v_1,v_2)

#sess = tf.Session() 
#print(ses.run(tv_add)) 
#sess.close()

#with tf.Session() as sess:
#     print(sess.run(message).decode())
#     print(sess.run(v_add))
     
sess = tf.InteractiveSession() 
print(v_add.eval()) 
sess.close()     