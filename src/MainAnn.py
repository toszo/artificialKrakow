import os.path
import pickle
import random

import gym

from ann import Ann

class MainAnn:
    def __init__(self, environmentName):
        self.episodeData = []
        self.episodeIndex = dict()
        self.environmentName = environmentName
        self.steps = 0
        self.trainEverySteps = 10
       
    def chooseRandomAction(self, qValues):
        return random.randint(0, len(qValues) - 1)

    def chooseBestAction(self, qValues):
        return qValues.index(max(qValues))

    def chooseRandomOrBest(self,qvalues):
        if(random.randint(0,100) > 90):
            return self.chooseRandomAction(qvalues)
        else:
            return self.chooseBestAction(qvalues)

    def execute(self):
        env = gym.make(self.environmentName)

        net = self.loadQ()
        q = Ann(env, self.environmentName)
        if net == None:            
            q.learnDefault()
        else:
            q.net = net
        self.clearCouterLog()

        iter=0;
        while True:
            observation = env.reset()
            steps = self.runEpisode(env, observation, q)
            iter+=1
            if(iter % 1000):
                self.saveNetworkData(q)

            
    def counterFileName(self):
        return self.environmentName + '.counter.dat'

    def clearCouterLog(self):
        file = open(self.counterFileName(),"w")
        file.seek(0)
        file.truncate()

    def logCounter(self,steps):
        file = open(self.counterFileName(),"a")
        file.write(str(steps)+"\n")


    def episodeDataFileName(self):
        return self.environmentName + '.episodeData.dat'

    def networkDataFileName(self):
        return self.environmentName + '.network.dat'

    def saveNetworkData(self,q):
        with open(self.networkDataFileName(), 'wb') as output:
            pickle.dump(q.net, output, pickle.HIGHEST_PROTOCOL)


    def saveEpisodeData(self):
        with open(self.episodeDataFileName(), 'wb') as output:
            pickle.dump(self.episodeData, output, pickle.HIGHEST_PROTOCOL)

    def loadQ(self):
        if not os.path.isfile(self.networkDataFileName()):
           return None
        else:
            with open(self.networkDataFileName(), 'rb') as input:
                return pickle.load(input)
          
    def runEpisode(self, env, observation, q):
        done = False
        self.steps =0
        while not done:
            qValues = q.calculate(observation)

            action = self.chooseRandomOrBest(qValues)

            newObservation, reward, done, info = env.step(action)

            episode = {'observation':observation, 'action':action, 'newObservation':newObservation, 'reward':reward, 'done':done}
            
            q.learn(episode)
            
            observation = newObservation
            env.render()
            self.steps += 1
            if self.steps % self.trainEverySteps == 0:
                q.train()                

        self.logCounter(self.steps)
        print('Steps:'+str(self.steps))
        return self.steps

    def learnFromPreviousExperience(self, q):
        for _ in range(len(self.episodeIndex.keys()) * 10):
            indexKeys = list(self.episodeIndex.keys())
            randomKey = indexKeys[random.randint(0, len(indexKeys)-1)]
            episodeList = self.episodeIndex[randomKey]
            episode = episodeList[random.randint(0, len(episodeList)-1)]
            q.learn(episode)
