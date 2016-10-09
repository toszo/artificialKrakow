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
        self.endingLength = 10

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
        self.loadEpisodeData(discreter)
        self.clearCouterLog()

        while True:
            allHistoricObservations = [episode['observation'] for episode in self.episodeData]
            withinLimits = discreter.observationsWithinLimits(allHistoricObservations)
            if not withinLimits:
                iteration = 0
                q = Q(env, self.environmentName)
                q.save()
                discreter.updateLimits(allHistoricObservations)
                discreter.save()

            self.runEpisode(env, q, discreter)
            self.learnFromPreviousExperience(q, discreter)
            

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


    def runEpisode(self, env, q, discreter):
        observation = env.reset()
        done = False
        while not done:
            state = discreter.getState(observation)
            qValues = q.calculate(state)

            action = self.chooseActionBasedOnIndex(qValues,state)

            newObservation, reward, done, info = env.step(action)

            newState = discreter.getState(newObservation)
            q.learn(state, action, newState, reward, done)
            self.addEpisode({'observation':observation, 'action':action, 'newObservation':newObservation, 'reward':reward, 'done':done}, discreter)

            observation = newObservation
            env.render()
            self.steps += 1
            if self.steps % self.saveEverySteps == 0:
                self.saveEpisodeData()
                print('Steps:'+str(self.steps)+'. Episode data saved.')
        
        self.saveEpisodeData()
        print('Steps:'+str(self.steps)+'(end of episode). Episode data saved.')
        self.logCounter(self.steps)
        

    def addEpisode(self, episode, discreter):
        self.episodeData.append(episode)
        self.updateEpisodeIndex(episode, discreter)
    def updateEpisodeIndex(self, episode, discreter):
        stateKey = str(discreter.getState(episode['observation']))
        if not stateKey in self.episodeIndex.keys():
            self.episodeIndex[stateKey] = []
        self.episodeIndex[stateKey].append(episode)


    def learnFromPreviousExperience(self, q, discreter):
        self.learnFromEndings(q, discreter)
        for _ in range(len(self.episodeIndex.keys()) * 1):
            indexKeys = list(self.episodeIndex.keys())
            randomKey = indexKeys[random.randint(0, len(indexKeys)-1)]
            episodeList = self.episodeIndex[randomKey]
            episode = episodeList[random.randint(0, len(episodeList)-1)]
            self.learnEpisode(episode, q, discreter)

    def learnEpisode(self, episode, q, discreter):
        q.learn(discreter.getState(episode['observation']), episode['action'], discreter.getState(episode['newObservation']), episode['reward'], episode['done'])

    def findEndings(self):
        endings = [{'end':e, 'index':i, 'prev':[]} for i, e in enumerate(self.episodeData) if e['done']]
        for index, ending in enumerate(endings):
            prev = ending['end']
            prev_index = ending['index']
            for _ in range(self.endingLength):
                if prev_index > 0 and (self.episodeData[prev_index-1]['newObservation'] == prev['observation']).all():
                    prev = self.episodeData[prev_index-1]
                    prev_index = prev_index - 1
                else:   
                    pre_prevs = [e for e in self.episodeData if (e['newObservation'] == prev['observation']).all()]
                    if len(pre_prevs) != 1:
                        break
                    prev = pre_prevs[0]
                ending['prev'].append(prev)
        return [[ending['end']] + ending['prev'] for ending in endings]        

    def learnFromEndings(self, q, discreter):
        endings = self.findEndings()
        print('Endings:', len(endings))
        episodes = [episode for ending in endings for episode in ending]
        for _ in range(10):
            for episode in episodes:
                self.learnEpisode(episode, q, discreter)
            
            
        
