#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug

import data_loader as data

tf.logging.set_verbosity(tf.logging.DEBUG)
FLAGS = None

def cnn_model_fn(features, labels, mode):
  img_size = 128
  num_filter = 16 #32
  filter_size = 2
  pool_size = 2

  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  input_layer = tf.reshape(features["x"], [-1, img_size, img_size, 1])

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=num_filter,
      kernel_size=[filter_size, filter_size],
      padding="same",
      activation=tf.nn.relu)
  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[pool_size, pool_size], strides=pool_size)
  current_img_size = img_size//pool_size

  # Convolutional Layer #2
  # Computes 32 features using a 5x5 filter.
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=num_filter*2,
      kernel_size=[filter_size+1, filter_size+1],
      padding="same",
      activation=tf.nn.relu)
    # Pooling Layer #2
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[pool_size, pool_size], strides=pool_size)
  current_img_size = img_size//pool_size

  # Convolutional Layer #2
  # Computes 32 features using a 5x5 filter.
  conv3 = tf.layers.conv2d(
      inputs=pool2,
      filters=num_filter,
      kernel_size=[filter_size+3, filter_size+3],
      padding="same",
      activation=tf.nn.relu)
  # Pooling Layer #2
  pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[pool_size, pool_size], strides=pool_size)
  current_img_size = current_img_size//pool_size

  # Flatten tensor into a batch of vectors
  flat_size = pool3.shape[1].value * pool3.shape[2].value * pool3.shape[3].value
  pool_flat = tf.reshape(pool3, [-1, flat_size])

  # Dense Layer
  dense = tf.layers.dense(inputs=pool_flat, units=flat_size, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.3, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  logits = tf.layers.dense(inputs=dropout, units=10)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)

  loss = tf.losses.softmax_cross_entropy(
      onehot_labels=onehot_labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(argv):
  if FLAGS.predict is None:
    train()
  else:
    predict()

def train():
  # Load training and eval data
  data_train_data, data_train_labels = data.get_train_data(onehot=False)
  train_data = np.asarray(data_train_data, dtype=np.float32)
  train_labels = np.asarray(data_train_labels, dtype=np.int32)
  data_eval_data, data_eval_labels = data.get_test_data(onehot=False)
  eval_data = np.asarray(data_eval_data, dtype=np.float32)
  eval_labels = np.asarray(data_eval_labels, dtype=np.int32)

  # Create the Estimator
  mnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir=FLAGS.model)

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)
  debug_hook = tf_debug.LocalCLIDebugHook()

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=10,
      num_epochs=None,
      shuffle=True)
  mnist_classifier.train(
      input_fn=train_input_fn,
      steps=FLAGS.step,
      hooks=[logging_hook])

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)

def predict():
  predict_data = np.asarray([data.get_img(FLAGS.predict)], dtype=np.float32)
  mnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir=FLAGS.model)
  input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": predict_data},
      num_epochs=1,
      shuffle=False)
  results = mnist_classifier.predict(input_fn=input_fn)
  for i, p in enumerate(results):
    print("Prediction probabilities: %s" % p)
    print("The number is %s." % p["classes"])


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--step', type=int, default=20000,
                      help='steps to train')
  parser.add_argument('--model', type=str, default='data/convnet_model',
                      help='folder to save model')
  parser.add_argument('--predict', type=str,
                      help='predict the picture, bmp bw 128x128')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
