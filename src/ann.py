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

class Ann:

    InputSpaceSize = 4
    OutputSpaceSize = 2
    InternalSpaceSize = 6
    def __init__(self):
        self.episodeData = []
        self.net = buildNetwork(InputSpaceSize, InternalSpaceSize, OutputSpaceSize)     
        self.ds = SupervisedDataSet(InputSpaceSize, OutputSpaceSize)
    
    fileName = 'episodeData.dat'

    def load(self):
        if not os.path.isfile(Ann.fileName):
            self.episodeData = []
        else:
            with open(Ann.fileName, 'rb') as input:
                self.episodeData = pickle.load(input)
    
    def train(self):
        trainer = BackpropTrainer(self.net, self.ds)
        trainer.trainUntilConvergence()

    def learn(self):
        self.load()

        for episode in self.episodeData:
             observation = [episode['observation'][i] for i in range(InputSpaceSize))]
             newObservation = [episode['newObservation'][i] for i in range(InputSpaceSize))]
             q1 = self.net.activate(observation)
             q2 = self.net.activate(newObservation)
             
       # self.train()

def main(args=None):
    Ann().learn()

if __name__ == "__main__":
    main()