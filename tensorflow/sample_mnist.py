#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf

import data_loader as data

FLAGS = None
IMG_SIZE = 16384


def main(_):
  # Create the model
  x = tf.placeholder(tf.float32, [None, IMG_SIZE])
  W = tf.Variable(tf.zeros([IMG_SIZE, 10]))
  b = tf.Variable(tf.zeros([10]))
  h = tf.matmul(x, W) + b

  y = tf.placeholder(tf.float32, [None, 10])

  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=h))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  # Train
  for _ in range(1000):
    batch_xs, batch_ys = data.get_train_data()
    rtn = sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(h, 1), tf.argmax(y, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  test_xs, test_ys = data.get_test_data()
  print("Accuracy: ", sess.run(accuracy, feed_dict={x: test_xs,
                                      y: test_ys}))

  img = data.get_img(FLAGS.ask)
  print("The image is: ", sess.run(tf.argmax(h, 1), feed_dict={x: [img]}))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='data',
                      help='Directory for storing input data')
  parser.add_argument('--ask', type=str, default='data/0x96f6.bmp',
                      help='the picture to classify, bmp bw 128x128')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
