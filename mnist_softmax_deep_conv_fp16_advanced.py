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


from pdb import set_trace as bp


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


def create_simple_model(nbatch, nin, nout, dtype):
  data = tf.placeholder(dtype, shape=(None, 784))
  image = tf.reshape(data, [-1, 28, 28, 1])

  conv1_weights = tf.get_variable(name='conv1w', initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=dtype),
                                  shape=[FLAGS.conv_size, FLAGS.conv_size, 1, FLAGS.conv_depth],
                                  dtype=dtype)
  conv1_biases = tf.get_variable('conv1b', (FLAGS.conv_depth), dtype,
                                 initializer=tf.zeros_initializer())
  conv2_weights = tf.get_variable(name='conv2w', initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=dtype),
                                  shape=[FLAGS.conv_size, FLAGS.conv_size, FLAGS.conv_depth, FLAGS.conv_depth],
                                  dtype=dtype)
  conv2_biases = tf.get_variable('conv2b', (FLAGS.conv_depth), dtype,
                                 initializer=tf.zeros_initializer())

  conv1 = tf.nn.conv2d(image, conv1_weights, [1, 1, 1, 1], padding='SAME')
  hidden1 = tf.nn.relu(conv1 + conv1_biases)
  conv2 = tf.nn.conv2d(hidden1, conv2_weights, [1, 1, 1, 1], padding='SAME')
  hidden2 = tf.nn.relu(conv2 + conv2_biases)

  hidden2_flat = tf.contrib.layers.flatten(hidden2)

  W1 = tf.get_variable('w1', (hidden2_flat.shape.as_list()[1], FLAGS.num_hunits), dtype)
  b1 = tf.get_variable('b1', (FLAGS.num_hunits), dtype, initializer=tf.zeros_initializer())
  z = tf.nn.relu(tf.matmul(hidden2_flat, W1) + b1)
  W2 = tf.get_variable('w2', (FLAGS.num_hunits, 10), dtype)
  b2 = tf.get_variable('b2', (10), dtype, initializer=tf.zeros_initializer())
  logits = tf.matmul(z, W2) + b2
  target = tf.placeholder(tf.int64, shape=(None))

  loss = tf.losses.sparse_softmax_cross_entropy(target, tf.cast(logits, tf.float32))
  return data, target, logits, loss


def main(_):
  nbatch = 64
  nin = 100
  nout = 10
  learning_rate = 0.1
  momentum = 0.9
  loss_scale = FLAGS.loss_scale
  dtype = tf.float16
  tf.set_random_seed(1234)

  # Create training graph
  with tf.device('/gpu:0'), \
       tf.variable_scope(
         # Note: This forces trainable variables to be stored as float32
         'fp32_storage', custom_getter=float32_variable_storage_getter):
    data, target, logits, loss = create_simple_model(nbatch, nin, nout, dtype)
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    # Note: Loss scaling can improve numerical stability for fp16 training
    grads = gradients_with_loss_scaling(loss, variables, loss_scale)
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
    training_step_op = optimizer.apply_gradients(zip(grads, variables))
    init_op = tf.global_variables_initializer()

  mnist = input_data.read_data_sets(FLAGS.data_dir)

  sess = tf.Session()
  sess.run(init_op)

  for step in range(6000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    np_loss, _ = sess.run([loss, training_step_op],
                          feed_dict={data: batch_xs, target: batch_ys})
    if step % 1000 == 0:
      print('%4i %6f' % (step + 1, np_loss))

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(logits, 1), target)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float16))
  print(sess.run(
      accuracy, feed_dict={
          data: mnist.test.images,
          target: mnist.test.labels
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
    default=1024,
    help='Number of units in the hidden layer'
  )
  parser.add_argument(
    '--conv_size',
    type=int,
    default=5,
    help='Size of convolution'
  )
  parser.add_argument(
    '--conv_depth',
    type=int,
    default=32,
    help='Depth of convolution'
  )
  parser.add_argument(
    '--loss_scale',
    type=int,
    default=128,
    help='Loss scale'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)