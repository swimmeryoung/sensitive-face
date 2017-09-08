# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse

import tensorflow as tf

FLAGS = None

def read_image(file_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(file_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
          'img_raw': tf.FixedLenFeature([], tf.string),
          'label': tf.FixedLenFeature([], tf.int64),
          })

    image = tf.decode_raw(features['img_raw'], tf.uint8)
    image = tf.reshape(image,[32,32,3])
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)
    return image, label

def read_image_batch(file_queue, batch_size):
    img, label = read_image(file_queue)
    capacity = 3 * batch_size
    image_batch, label_batch = tf.train.batch([img, label], batch_size=batch_size, capacity=capacity, num_threads=10)
    one_hot_labels = tf.to_float(tf.one_hot(label_batch, 2, 1, 0))
    return image_batch, one_hot_labels

def deepnn(x):
    """deepnn builds the graph for a deep net for classifying digits.
    Args:
      x: an input tensor with the dimensions (N_examples, 784), where 784 is the
      number of pixels in a standard MNIST image.
    Returns:
      A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
      equal to the logits of classifying the digit into one of 10 classes (the
      digits 0-9). keep_prob is a scalar placeholder for the probability of
      dropout.
    """
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    x_image = tf.reshape(x, [-1, 32, 32, 3])

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    W_conv1 = weight_variable([5, 5, 3, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # Pooling layer - downsamples by 2X.
    h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer -- maps 32 feature maps to 64.
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Second pooling layer.
    h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    W_fc1 = weight_variable([8 * 8 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1024 features to 10 classes, one for each digit
    W_fc2 = weight_variable([1024, 2])
    b_fc2 = bias_variable([2])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv, keep_prob


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)




def main(_):
    #训练数据集
    train_file_path = os.path.join(FLAGS.buckets, "face_train.tfrecords")
    #测试数据集
    test_file_path = os.path.join(FLAGS.buckets, "face_test.tfrecords")
    #模型存储名称
    ckpt_path = os.path.join(FLAGS.checkpointDir, "model.ckpt")
    train_image_filename_queue = tf.train.string_input_producer(
            tf.train.match_filenames_once(train_file_path))
    train_images, train_labels = read_image_batch(train_image_filename_queue,4000)
    test_image_filename_queue = tf.train.string_input_producer(
            tf.train.match_filenames_once(test_file_path))
    test_images, test_labels = read_image_batch(test_image_filename_queue, 1500)
    
    train_labels = tf.to_float(train_labels)
    test_labels = tf.to_float(test_labels)

    # create the model
    x = tf.placeholder(tf.float32, [None, 32,32,3])
    y_ = tf.placeholder(tf.float32,[None, 2])
    y_conv, keep_prob = deepnn(x)

    cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        train_images = sess.run(train_images)
        train_labels = sess.run(tf.to_float(train_labels))
        test_images = sess.run(test_images)
        test_labels = sess.run(tf.to_float(test_labels))

        for i in range(1000):
            
            if ((i+1)%100==0):
                train_accuracy = accuracy.eval(feed_dict={x:train_images, y_:train_labels,keep_prob:0.5})
                print('step %d, train_accuracy %g' % (i+1,train_accuracy))
            sess.run(train_step,feed_dict={x:train_images, y_:train_labels,keep_prob:1})
        print('test accuracy %g' % accuracy.eval(feed_dict={x:test_images, y_:test_labels,keep_prob:1.0}))
    
        saver = tf.train.Saver()

        save_path = saver.save(sess, ckpt_path)
        print("Model saved in file: %s" % save_path)

        # stop queue runner
        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #获得buckets路径
    parser.add_argument('--buckets', type=str, default='',
                        help='input data path')
    #获得checkpoint路径
    parser.add_argument('--checkpointDir', type=str, default='',
                        help='output model path')
    FLAGS, _ = parser.parse_known_args()
    tf.app.run(main=main)

