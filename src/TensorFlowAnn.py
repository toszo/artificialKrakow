import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import math
import os.path

class Network:
    pass

class TensorFlowAnn:
    def __init__(self, environmentName, actions, hiddenLayerSizes):
        self.fileName = environmentName + '.network.dat'
        self.createNetworks(hiddenLayerSizes, actions)
        
    def initWeights(self, shape, name):
        return tf.Variable(tf.random_normal(shape, stddev=1/np.sqrt(shape[0])), name=name)

    def createNetwork(self, sizes, variablePrefix):
        x = tf.placeholder(tf.float32, [None, sizes[0]], name='{0}_x'.format(variablePrefix))
        y = tf.placeholder(tf.float32, [None, sizes[-1]], name='{0}_y'.format(variablePrefix))

        a = x
        layer = 1
        for sizein, sizeout in zip(sizes[:-2], sizes[1:-1]):
            W = self.initWeights([sizein, sizeout], name='{0}_W{1}'.format(variablePrefix, layer))
            b = self.initWeights([sizeout], name='{0}_b{1}'.format(variablePrefix, layer))
            a = tf.nn.relu(tf.matmul(a, W) + b)
            layer += 1
        
        sizein = sizes[-2]
        sizeout = sizes[-1]
        W = self.initWeights([sizein, sizeout], name='{0}_W{1}'.format(variablePrefix, layer))
        b = self.initWeights([sizeout], name='{0}_b{1}'.format(variablePrefix, layer))
        a = tf.matmul(a, W) + b

        cost = tf.reduce_mean(tf.pow(tf.sub(a, y), 2))
        
        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
        
        network = Network()
        network.x = x
        network.y = y
        network.a = a
        network.cost = cost
        network.train_step = train_step
        return network

    def createNetworks(self, sizes, actions):
        self.Qs = []
        self.Ss = []
        self.Rs = []
        for action in range(0, actions):
            self.Qs.append(self.createNetwork(sizes+[1], 'Q'+str(action)))
            self.Ss.append(self.createNetwork(sizes+[sizes[0]], 'S'+str(action)))
            self.Rs.append(self.createNetwork(sizes+[1], 'R'+str(action)))

        self.session = tf.Session()

        if os.path.isfile(self.fileName):
            saver = tf.train.Saver()
            saver.restore(self.session, self.fileName)
            print('loaded networks')
        else:
            self.session.run(tf.initialize_all_variables())
            print('created new networks')

    def trainQs(self, action, trX, trY):
        network = self.Qs[action]
        self.session.run(network.train_step, feed_dict={network.x: trX, network.y: trY})

    def trainSs(self, action, trX, trY):
        network = self.Ss[action]
        self.session.run(network.train_step, feed_dict={network.x: trX, network.y: trY})

    def trainRs(self, action, trX, trY):
        network = self.Rs[action]
        self.session.run(network.train_step, feed_dict={network.x: trX, network.y: trY})

    def saveNetworks(self):
        saver = tf.train.Saver()
        saver.save(self.session, self.fileName)
        print('saved networks')

    def calculateBatch(self, ann, inX):
        out = np.array([self.session.run(network.a, feed_dict={network.x: inX}) for network in ann])
        out = np.transpose(out, axes=[1,0,2])
        return out
            
            
        
