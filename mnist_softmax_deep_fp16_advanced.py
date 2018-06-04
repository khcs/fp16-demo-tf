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


def float32_variable_storage_getter(getter, name, shape=None, dtype=None,
                                    initializer=None, regularizer=None,
                                    trainable=True,
                                    *args, **kwargs):
  """Custom variable getter that forces trainable variables to be stored in
  float32 precision and then casts them to the training precision.
  """
  storage_dtype = tf.float32 if trainable else dtype
  variable = getter(name, shape, dtype=storage_dtype,
                    initializer=initializer, regularizer=regularizer,
                    trainable=trainable,
                    *args, **kwargs)
  if trainable and dtype != tf.float32:
    variable = tf.cast(variable, dtype)
  return variable


def gradients_with_loss_scaling(loss, variables, loss_scale):
    """Gradient calculation with loss scaling to improve numerical stability
    when training with float16.
    """
    return [grad / loss_scale
            for grad in tf.gradients(loss * loss_scale, variables)]


def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir)

  x = tf.placeholder(tf.float16, [None, 784]) #### FP16 ####
  with tf.variable_scope('fp32_vars', custom_getter=float32_variable_storage_getter):
    # Create the model
    W1 = tf.Variable(tf.truncated_normal([784, FLAGS.num_hunits], dtype=tf.float16)) #### FP16 ####
    b1 = tf.Variable(tf.zeros([FLAGS.num_hunits], dtype=tf.float16)) #### FP16 ####
    z = tf.nn.relu(tf.matmul(x, W1) + b1)
    W2 = tf.Variable(tf.truncated_normal([FLAGS.num_hunits, 10], dtype=tf.float16))
    b2 = tf.Variable(tf.zeros([10], dtype=tf.float16))

    y = tf.matmul(z, W2) + b2

  y = tf.cast(y, tf.float32)

  # Define loss and optimize
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
  #train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  loss_scale = FLAGS.loss_scale
  variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
  grads = gradients_with_loss_scaling(cross_entropy, variables, loss_scale)
  grads, _ = tf.clip_by_global_norm(grads, 5.0)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
  train_step = optimizer.apply_gradients(zip(grads, variables))

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  # Train
  for _ in range(6000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), y_)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float16)) #### FP16 ####
  print(sess.run(
      accuracy, feed_dict={
          x: mnist.test.images,
          y_: mnist.test.labels
      }))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_dir',
      type=str,
      default='/tmp/tensorflow/mnist/input_data',
      help='Directory for storing input data')
  parser.add_argument(
    '--num_hunits',
    type=int,
    default=10,
    help='Number of units in the hidden layer'
  )
  parser.add_argument(
    '--loss_scale',
    type=int,
    default=128,
    help='Loss scale'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)