#!/usr/bin/python3
# -*- coding : utf-8 -*-

'''
    Author: Vinícius Matheus
    Github: Vnicius

    Based int the tutorial video: https://www.youtube.com/watch?v=BhpvH5DuVu8

    The code get the dataset mnist of TensorFlow to train a Neural Network 
    with dimensions defined by parameters.
'''

import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 28 x 28 images = 784 values
QUANT_DATA = 784
N_OUTPUT = 10

def load_data():
    mnist = input_data.read_data_sets('/tmp/data', one_hot=True)

    return mnist

def nn_model(data, number_hidden_layers, number_nodes):
    if not number_hidden_layers:
        sys.exit('The number of layers must to be more than 0!')

    elif not number_nodes:
        sys.exit('The number of nodes must to be more than 0!')

    # First layer = QUANT_DATA -> number_nodes
    hidden_layers_values = []   # values of weigths and biases of each hidden layer
    hidden_layers = []

    layer_1 = {'weigths' : tf.Variable(tf.random_normal([QUANT_DATA, number_nodes])),
               'biases' : tf.Variable(tf.random_normal([number_nodes]))}

    hidden_layers_values.append(layer_1)

    # Another layers = number_nodes -> number_nodes
    for _ in range(number_hidden_layers - 1):
        layer = {'weigths' : tf.Variable(tf.random_normal([number_nodes, number_nodes])),
                 'biases' : tf.Variable(tf.random_normal([number_nodes]))}

        hidden_layers_values.append(layer)

    # Output layer = number_nodes -> N_OUTPUT
    out_layer = {'weigths' : tf.Variable(tf.random_normal([number_nodes, N_OUTPUT])),
                 'biases' : tf.Variable(tf.random_normal([N_OUTPUT]))}

    # (input_data * weigths) + biases

    # First layer = 784 -> number_nodes
    layer_1 = tf.add(tf.matmul(data, hidden_layers_values[0]['weigths']),
                     hidden_layers_values[0]['biases'])
    layer_1 = tf.nn.relu(layer_1)   # activation function

    hidden_layers.append(layer_1)

    for i in range(1, number_hidden_layers - 1):
        layer = tf.add(tf.matmul(hidden_layers[i-1], hidden_layers_values[i]['weigths']),
                       hidden_layers_values[i]['biases'])
        layer = tf.nn.relu(layer)

        hidden_layers.append(layer)

    # Output layer = number_nodes -> N_OUTPUT
    output = tf.add(tf.matmul(hidden_layers[-1], out_layer['weigths']),
                    out_layer['biases'])

    return output

def train_nn(x, y, number_hidden_layers, number_nodes, number_epochs, batch_size):
    prediction = nn_model(x, number_hidden_layers, number_nodes)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    epochs = number_epochs
    mnist = load_data()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                current_x, current_y = mnist.train.next_batch(batch_size)
                _, epoch_cost = sess.run([optimizer, cost],
                                         feed_dict={x : current_x, y : current_y})

                loss += epoch_cost

            print("\nEpoch: " + str(epoch + 1) + " of " + str(number_epochs) + "\nLoss: " + str(loss))

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        print("\nAccuracy: " + str(accuracy.eval({x : mnist.test.images, y: mnist.test.labels}) * 100) + "%")

if __name__ == '__main__':

    num_layers = int(sys.argv[1])
    num_nodes = int(sys.argv[2])
    num_epochs = int(sys.argv[3])
    batch_size = int(sys.argv[4])
    

    x = tf.placeholder('float', [None, QUANT_DATA])
    y = tf.placeholder('float', [None, N_OUTPUT])

    train_nn(x, y, num_layers, num_nodes, num_epochs, batch_size)