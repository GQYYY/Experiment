#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf
import numpy as np

def prob_filter(sorted_probs,threshold):
    i = tf.constant(-2)
    prob_sum = tf.constant(0.0.dtype=tf.float32)

    condition = lambda i,j: tf.less(j,threshold)
    body = lambda i,j; (i+1,tf.reduce_sum(tf.slice(a,[0],[i+2])))
    result = tf.while_loop(condition,body,[i,prob_sum])

    return result
