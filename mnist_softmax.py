# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A very simple MNIST classifier.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir)
  # log directory
  
  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])  # unlimited images with 28*28 pixel each
  W = tf.Variable(tf.zeros([784, 10])) # 28*28 image for each row from 0 to 9 
  b = tf.Variable(tf.zeros([10])) 
  y = tf.matmul(x, W) + b  # y=W*x+b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.int64, [None])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.losses.sparse_softmax_cross_entropy on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  # Train
  train_writer = tf.summary.FileWriter( "C:/Users/handa_k/AppData/Local/Temp/mnist", sess.graph)
  counter=0
  for _ in range(1000):
    counter+=1
    merge = tf.summary.merge_all()
    summary=sess.run(merge)
    train_writer.add_summary(summary,counter)
    batch_xs, batch_ys = mnist.train.next_batch(100)  # next_batch is a method of the DataSet class, mnist.train is an instance of class DataSet
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys}) # batch_xs, is the number of images,batch_ys is the label  
	# merging for board
    

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), y_) 
  # did not understand this. Argmax calculate the index of maximum value and it can be compared with y_  ?
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  # histogram
  tf.summary.histogram("Correct Predictions", correct_prediction)
  print(sess.run(
      accuracy, feed_dict={
          x: mnist.test.images,    # what kind of instances are mnist.test.iamges and mnist.tets.labels and to which class it belongs to,
                                     # what exactly the meaning here.		  
          y_: mnist.test.labels
      }))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_dir',
      type=str,
      default='/tmp/tensorflow/mnist/input_data',
      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
