import os.path
import pickle
import random

import gym

from StateMapper import StateMapper
from Q import Q


class MainClass:
    def __init__(self, environmentName):
        self.episodeData = []
        self.episodeIndex = dict()
        self.environmentName = environmentName
        self.steps = 0
        self.saveEverySteps = 1000

    def chooseRandomAction(self, qValues):
        return random.randint(0, len(qValues) - 1)

    def chooseBestAction(self, qValues):
        return qValues.index(max(qValues))

    def chooseActionBasedOnIndex(self,qValues,state):
        indexValues = [len(value) for value in self.episodeIndex.values()]

        valuesCount = len(indexValues)

        if (valuesCount != 0 and str(state) in self.episodeIndex.keys() and len(self.episodeIndex[str(state)]) > sum(indexValues)/valuesCount):
            action = self.chooseBestAction(qValues)
        else:
            action = self.chooseRandomAction(qValues)
        return action

    def execute(self):
        env = gym.make(self.environmentName)
        q = Q.load(env, self.environmentName)
        discreter = StateMapper.load(len(env.observation_space.high), self.environmentName)
        print('StateMapper configuration loaded')
        print('  high:'+str(discreter.high))
        print('   low:'+str(discreter.low))
        self.loadEpisodeData(discreter)
        self.clearCouterLog()

        while True:
            observation = env.reset()
            self.runEpisode(env, observation, q, discreter)
            self.learnFromPreviousExperience(q, discreter)
            allHistoricObservations = [episode['observation'] for episode in self.episodeData]
            changed = discreter.update(allHistoricObservations)
            if changed:
                iteration = 0
                q = Q(env, self.environmentName)
                q.save()
                discreter.extendVector()
                discreter.save()
                print('New StateMapper configuration saved')
                print('  high:'+str(discreter.high))
                print('   low:'+str(discreter.low))

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
        
    def saveEpisodeData(self):
        with open(self.episodeDataFileName(), 'wb') as output:
            pickle.dump(self.episodeData, output, pickle.HIGHEST_PROTOCOL)

    def loadEpisodeData(self, discreter):
        if not os.path.isfile(self.episodeDataFileName()):
            self.episodeData = []
        else:
            with open(self.episodeDataFileName(), 'rb') as input:
                self.episodeData = pickle.load(input)
        for episode in self.episodeData:
            self.updateEpisodeIndex(episode, discreter)


    def runEpisode(self, env, observation, q, discreter):
        done = False
        while not done:
            state = discreter.getState(observation)
            qValues = q.calculate(state)

            action = self.chooseActionBasedOnIndex(qValues,state)

            newObservation, reward, done, info = env.step(action)

            newState = discreter.getState(newObservation)
            q.learn(state, action, newState, reward, done)
            self.saveEpisode({'observation':observation, 'action':action, 'newObservation':newObservation, 'reward':reward, 'done':done}, discreter)

            observation = newObservation
            env.render()
            self.steps += 1
            if self.steps % self.saveEverySteps == 0:
                self.saveEpisodeData()
                print('Steps:'+str(self.steps)+'. Episode data saved.')

        self.logCounter(self.steps)
        

    def saveEpisode(self, episode, discreter):
        self.episodeData.append(episode)
        self.updateEpisodeIndex(episode, discreter)
    def updateEpisodeIndex(self, episode, discreter):
        stateKey = str(discreter.getState(episode['observation']))
        if not stateKey in self.episodeIndex.keys():
            self.episodeIndex[stateKey] = []
        self.episodeIndex[stateKey].append(episode)


    def learnFromPreviousExperience(self, q, discreter):
        for _ in range(len(self.episodeIndex.keys()) * 10):
            indexKeys = list(self.episodeIndex.keys())
            randomKey = indexKeys[random.randint(0, len(indexKeys)-1)]
            episodeList = self.episodeIndex[randomKey]
            episode = episodeList[random.randint(0, len(episodeList)-1)]
            q.learn(discreter.getState(episode['observation']), episode['action'], discreter.getState(episode['newObservation']), episode['reward'], episode['done'])
