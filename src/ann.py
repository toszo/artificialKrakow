# How to get this shit.
# git clone git://github.com/pybrain/pybrain.git
# sudo python3 setup.py install
# sudo apt-get install python3-numpy python3-scipy
# Should be enough.

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
import os.path
import pickle
import gym

class Ann:    
    DiscountFactor = 0.9
    def __init__(self, env, environmentName):
        
        self.episodeData = []
    
        self.inputSpaceSize = env.observation_space.shape[0]
        self.outputSpaceSize = env.action_space.n
        self.internalSpaceSize = 3

        self.net = buildNetwork(self.inputSpaceSize,self.internalSpaceSize, self.outputSpaceSize)     
        self.ds = SupervisedDataSet(self.inputSpaceSize, self.outputSpaceSize)

        self.fileName = environmentName+'.episodeData.dat'

    def getLearnedValue(self, qValues, reward):
        return reward + Ann.DiscountFactor * max(qValues)

    def load(self):
        if not os.path.isfile(self.fileName):
            self.episodeData = []
        else:
            with open(self.fileName, 'rb') as input:
                self.episodeData = pickle.load(input)
    
    def train(self):
        print("Network training - samples:"+str(len(self.ds)))
        trainer = BackpropTrainer(self.net, self.ds)
        trainer.trainUntilConvergence(maxEpochs=500)
        self.ds.clear()


    def learn(self,episode):
        observation = [episode['observation'][i] for i in range(self.inputSpaceSize)]
        newObservation = [episode['newObservation'][i] for i in range(self.inputSpaceSize)]
        reward = episode['reward']
        action = episode['action']
        q = self.calculate(observation)
        q2 = self.calculate(newObservation)
        if episode['done'] :
            learned_q = 0
            self.ds.addSample(newObservation,[0]*self.outputSpaceSize)    
        else:   
            learned_q = self.getLearnedValue(q2,reward)
        q[action] = learned_q
        self.ds.addSample(observation,q)

    def calculate(self,observation):
        return list(self.net.activate(observation))

    SampleSize = 2 # in percentege
    def learnDefault(self):
        self.load()

        idx = 0
        maxIdx = len(self.episodeData)-1

        while idx < maxIdx:
            self.learn(self.episodeData[idx])
            idx+= int(100/Ann.SampleSize)

        if(len(self.ds) >= self.inputSpaceSize):
            self.train()
        
# def main(args=None):
#     env = gym.make('CartPole-v0')
#     Ann(env,'CartPole-v0').learnDefault()

# if __name__ == "__main__":
#     main()