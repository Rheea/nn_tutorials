#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 14:55:42 2018

@author: aradulescu
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as pls
import matplotlib.cm as cm
import matplotlib.patches as mpatches

def tf_reset():
    try:
        sess.close()
    except:
        pass
    tf.reset_default_graph()
    return tf.Session()


sess = tf_reset()

a = tf.constant(1.0)
b = tf.constant(2.0)
c = a + b
c_run = sess.run(c)

print('c = {0}'.format(c_run))