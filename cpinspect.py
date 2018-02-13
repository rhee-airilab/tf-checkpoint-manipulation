#!/usr/bin/env python
# coding: utf-8
# cpinspect.py
from __future__ import print_function, division, absolute_import
import os
import sys
import tensorflow as tf

def cpinspect(cp_or_dir):
    #l = tf.train.list_variables(cp_or_dir) # return list of (name, shape) tuples
    l = tf.contrib.framework.list_variables(cp_or_dir) # return list of (name, shape) tuples
    return l

if __name__ == '__main__':
    l = cpinspect(sys.argv[1])
    for name, shape in l:
        #print("'{}', # shape: {}".format(name,shape))
        print("{} {}".format(name,shape))
