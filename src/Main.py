import os.path
import pickle
import random

import gym

from StateMapper import StateMapper
from Q import Q


class MainClass:
    def __init__(self, environmentName):
        self.stepsData = []
        self.stepsIndex = dict()
        self.environmentName = environmentName
        self.stepCount = 0
        self.saveStepsAfter = 1000
        self.endingLength = 10

    def chooseRandomAction(self, qValues):
        return random.randint(0, len(qValues) - 1)

    def chooseBestAction(self, qValues):
        return qValues.index(max(qValues))

    def chooseActionBasedOnIndex(self,qValues,state):
        indexValues = [len(value) for value in self.stepsIndex.values()]

        valuesCount = len(indexValues)

        if (valuesCount != 0 and str(state) in self.stepsIndex.keys() and len(self.stepsIndex[str(state)]) > sum(indexValues)/valuesCount):
            action = self.chooseBestAction(qValues)
        else:
            action = self.chooseRandomAction(qValues)
        return action

    def execute(self):
        env = gym.make(self.environmentName)
        q = Q.load(env, self.environmentName)
        discreter = StateMapper.load(len(env.observation_space.high), self.environmentName)
        self.loadStepsData(discreter)
        self.clearCouterLog()

        while True:
            allHistoricObservations = [step['observation'] for step in self.stepsData]
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


    def stepsDataFileName(self):
        return self.environmentName + '.stepsData.dat'
        
    def saveStepsData(self):
        with open(self.stepsDataFileName(), 'wb') as output:
            pickle.dump(self.stepsData, output, pickle.HIGHEST_PROTOCOL)

    def loadStepsData(self, discreter):
        if not os.path.isfile(self.stepsDataFileName()):
            self.stepsData = []
        else:
            with open(self.stepsDataFileName(), 'rb') as input:
                self.stepsData = pickle.load(input)
        for step in self.stepsData: 
            self.updateStepsIndex(step, discreter)


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
            self.addStep({'observation':observation, 'action':action, 'newObservation':newObservation, 'reward':reward, 'done':done}, discreter)

            observation = newObservation
            env.render()
            self.stepCount += 1
            if self.stepCount % self.saveStepsAfter == 0:
                self.saveStepsData()
                print('Steps performed:'+str(self.stepCount)+'. Steps data saved.')
        
        self.saveStepsData()
        print('Steps performed:'+str(self.stepCount)+'(end of episode). Steps data saved.')
        self.logCounter(self.stepCount)
        

    def addStep(self, step, discreter):
        self.stepsData.append(step)
        self.updateStepsIndex(step, discreter)
    def updateStepsIndex(self, step, discreter):
        stateKey = str(discreter.getState(step['observation']))
        if not stateKey in self.stepsIndex.keys():
            self.stepsIndex[stateKey] = []
        self.stepsIndex[stateKey].append(step)


    def learnFromPreviousExperience(self, q, discreter):
        self.learnFromEndings(q, discreter)
        for _ in range(len(self.stepsIndex.keys()) * 1):
            indexKeys = list(self.stepsIndex.keys())
            randomKey = indexKeys[random.randint(0, len(indexKeys)-1)]
            stepsList = self.stepsIndex[randomKey]
            step = stepsList[random.randint(0, len(stepsList)-1)]
            self.learnStep(step, q, discreter)

    def learnStep(self, step, q, discreter):
        q.learn(discreter.getState(step['observation']), step['action'], discreter.getState(step['newObservation']), step['reward'], step['done'])

    def findEndings(self):
        endings = [{'end':e, 'index':i, 'prev':[]} for i, e in enumerate(self.stepsData) if e['done']]
        for index, ending in enumerate(endings):
            prev = ending['end']
            prev_index = ending['index']
            for _ in range(self.endingLength):
                if prev_index > 0 and (self.stepsData[prev_index-1]['newObservation'] == prev['observation']).all():
                    prev = self.stepsData[prev_index-1]
                    prev_index = prev_index - 1
                else:   
                    pre_prevs = [e for e in self.stepsData if (e['newObservation'] == prev['observation']).all()]
                    if len(pre_prevs) != 1:
                        break
                    prev = pre_prevs[0]
                ending['prev'].append(prev)
        return [[ending['end']] + ending['prev'] for ending in endings]        

    def learnFromEndings(self, q, discreter):
        endings = self.findEndings()
        print('Endings:', len(endings))
        steps = [step for ending in endings for step in ending]
        for _ in range(10):
            for step in steps:
                self.learnStep(step, q, discreter)
            
        
