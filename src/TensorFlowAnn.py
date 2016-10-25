import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import math
import os.path

class TensorFlowAnn:
    def __init__(self, environmentName, sizes):
        self.fileName = environmentName + '.network.dat'
        (self.x, self.y, self.a_L, self.cost, self.train_step, self.session) = self.createAnn(sizes)
        
    def initWeights(self, shape):
        return tf.Variable(tf.random_normal(shape, stddev=1/np.sqrt(shape[0])))

    def createAnn(self, sizes):
        x = tf.placeholder(tf.float32, [None, sizes[0]])
        y = tf.placeholder(tf.float32, [None, sizes[-1]])

        a = x
        for sizein, sizeout in zip(sizes[:-2], sizes[1:-1]):
            W = self.initWeights([sizein, sizeout])
            b = self.initWeights([sizeout])
            a = tf.nn.relu(tf.matmul(a, W) + b)
        
        sizein = sizes[-2]
        sizeout = sizes[-1]
        W = self.initWeights([sizein, sizeout])
        b = self.initWeights([sizeout])
        a_L = tf.matmul(a, W) + b

        cost = tf.reduce_mean(tf.pow(tf.sub(a_L, y), 2))
        
        train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

        session = tf.Session()

        if os.path.isfile(self.fileName):
            saver = tf.train.Saver()
            saver.restore(session, self.fileName)
            print('loaded ann')
        else:
            session.run(tf.initialize_all_variables())
            print('created new ann')

        return (x, y, a_L, cost, train_step, session)

    def train(self, trX, trY):
        costs = []
        for i in range(20):
            self.session.run(self.train_step, feed_dict={self.x: trX, self.y: trY})
            cost = self.session.run(self.cost, feed_dict={self.x: trX, self.y: trY})
            print(i, 'cost', cost)
            #print('trY', trY[-1])
            #print('a_L', self.session.run(self.a_L, feed_dict={self.x: [trX[-1]]})[0])
            costs.append(math.log10(cost))

        saver = tf.train.Saver()
        saver.save(self.session, self.fileName)
        print('saved ann')

        # with plt.style.context('fivethirtyeight'):
        #     plt.plot(range(0, len(costs)), costs)
        # plt.show()   

    def calculate(self, inX):
        outY = self.calculateBatch([inX])
        return outY[0]

    def calculateBatch(self, inX):
        return self.session.run(self.a_L, feed_dict={self.x: inX})



def main(args=None):
    trX = np.array([[2,1], [4,2], [1,2], [2,4]])
    trY = np.array([[0.5], [0.5], [-0.5], [-0.5]])
    ann = TensorFlowAnn('TensorFlowAnn', [2, 10, 1])
    ann.train(trX, trY)
    print([ann.calculate(x) for x in trX])
 


if __name__ == "__main__":
    main()