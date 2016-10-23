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
        xSize = sizes[0]
        hSize = sizes[1]
        ySize = sizes[2]

        x = tf.placeholder(tf.float32, [None, xSize])
        y = tf.placeholder(tf.float32, [None, ySize])
        
        W_1 = self.initWeights([xSize, hSize])
        b_1 = self.initWeights([hSize])
        a_1 = tf.nn.sigmoid(tf.matmul(x, W_1) + b_1)

        W_L = self.initWeights([hSize, ySize])
        b_L = self.initWeights([ySize])
        #a_L = tf.nn.tanh(tf.matmul(a_1, W_L) + b_L)
        a_L = tf.matmul(a_1, W_L) + b_L

        cost = tf.reduce_sum(tf.pow(tf.sub(a_L, y), 2))
        
        train_step = tf.train.GradientDescentOptimizer(0.2).minimize(cost)

        session = tf.Session()

        if os.path.isfile(self.fileName):
            saver = tf.train.Saver()
            saver.restore(session, self.fileName)
        else:
            session.run(tf.initialize_all_variables())

        return (x, y, a_L, cost, train_step, session)

    def train(self, trX, trY):
        costs = []
        for i in range(10):
            self.session.run(self.train_step, feed_dict={self.x: trX, self.y: trY})
            cost = self.session.run(self.cost, feed_dict={self.x: trX, self.y: trY})
            costs.append(math.log10(cost))

        saver = tf.train.Saver()
        saver.save(self.session, self.fileName)

        # with plt.style.context('fivethirtyeight'):
        #     plt.plot(range(0, len(costs)), costs)
        # plt.show()   

    def calculate(self, inX):
        outY = self.session.run(self.a_L, feed_dict={self.x: [inX]})
        return list(outY[0])



def main(args=None):
    trX = np.array([[2,1], [4,2], [1,2], [2,4]])
    trY = np.array([[0.5, 0.1], [0.5, 0.2], [-0.5, 0.2], [-0.5, 0.4]])
    ann = TensorFlowAnn('TensorFlowAnn', [2, 10, 2])
    ann.train(trX, trY)
    print([ann.calculate(x) for x in trX])
 


if __name__ == "__main__":
    main()