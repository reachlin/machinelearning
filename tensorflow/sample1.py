#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

def main(args):
  # tensor is just matrix
  node1 = tf.constant(3.0, dtype=tf.float32)
  node2 = tf.constant(4)
  node3 = tf.constant([1,2,3], dtype=tf.int8)
  node4 = tf.constant([[1,2,3]], dtype=tf.int8)
  
  # this only prints tensor info.
  print(node1, node2, node3, node4)
  
  # this prints tensor values.
  sess = tf.Session()
  print(sess.run([node1, node2, node3]))

  # add node
  node_add = tf.add(node1, node1)
  print("node_add: ", node_add)
  print("sess.run(node_add): ",sess.run(node_add))

  # matrix multipy
  a = tf.placeholder(tf.float32)
  b = tf.placeholder(tf.float32)
  cross_node = tf.matmul(a, b)
  print(sess.run(cross_node, {a: [[1, 2, 3]], b: [[2, 4], [1, 3], [9, 2]]}))


if __name__ == "__main__":
  # tensorflow starts app from tf.app.run()
  # it will can main() in the package
  tf.app.run()
