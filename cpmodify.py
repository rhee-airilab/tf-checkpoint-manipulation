#!/usr/bin/env python
# coding: utf-8
# cpmodify.py
from __future__ import print_function, division, absolute_import
import os
import sys
import re
import tensorflow as tf

def cpmodify(cp_or_dir,from_regex,to_str,output_path,verbose=False):
    vars = tf.contrib.framework.list_variables(cp_or_dir) # return list of (name, shape) tuples
    session_config = tf.ConfigProto(
        allow_soft_placement=True,
        gpu_options={
            'allow_growth':True,
            'per_process_gpu_memory_fraction':0.0,
            })
    with tf.Graph().as_default(), tf.Session(config=session_config).as_default() as sess:
        new_vars = []
        for name, shape in vars:
            v = tf.contrib.framework.load_variable(cp_or_dir, name)
            new_name = re.sub(from_regex, to_str, name)
            new_var = tf.Variable(v, name=new_name)
            sess.run(new_var.initializer)
            new_vars.append(new_var)
            if verbose:
                print('# added:', new_name, name)
        saver = tf.train.Saver(new_vars)
        # sess.run(tf.global_variables_initializer())
        saver.save(sess,output_path,latest_filename='temp_cp')
        if verbose:
            for v in new_vars:
                print("'{}', # shape: {}".format(v.name,v.shape))
            

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print('Usage: cpmodify.py <cp-or-save-dir> <from-regex> <to-str> <output-path>',file=sys.stderr)
        sys.exit(1)
    cp_or_dir = sys.argv[1]
    from_regex = sys.argv[2]
    to_str = sys.argv[3]
    output_path = sys.argv[4]
    cpmodify(cp_or_dir, from_regex, to_str, output_path, verbose=True)