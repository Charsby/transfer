import os
import math
import random

import numpy as np
import tensorflow as tf
import cv2

slim = tf.contrib.slim

def rename(checkpoint_dir, replace_from, replace_to, add_prefix, dry_run,save_dir):
    "A function that can rename keys in saved tf model checkpoint file"
    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
    with tf.Graph().as_default():
        with tf.Session() as sess:
            for var_name, _ in tf.contrib.framework.list_variables(checkpoint_dir):
                # Load the variable
                var = tf.contrib.framework.load_variable(checkpoint_dir, var_name)
                # Set the new name
                new_name = var_name
                if None not in [replace_from, replace_to]:
                    new_name = new_name.replace(replace_from, replace_to)
                if add_prefix:
                    new_name = add_prefix + new_name

                if dry_run:
                    print('%s would be renamed to %s.' % (var_name, new_name))
                else:
                    print('Renaming %s to %s.' % (var_name, new_name))
                    # Rename the variable
                    var = tf.Variable(var, name=new_name)

            if not dry_run:
                # Save the variables
                saver = tf.train.Saver()
                sess.run(tf.global_variables_initializer())
                print "start saving"
                saver.save(sess,save_dir)
                print "all done"