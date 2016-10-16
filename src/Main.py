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
        stateMapper = StateMapper.load(len(env.observation_space.high), self.environmentName)
        self.loadStepsData(stateMapper)
        self.clearCouterLog()

        while True:
            allHistoricObservations = [step['observation'] for step in self.stepsData]
            withinLimits = stateMapper.observationsWithinLimits(allHistoricObservations)
            if not withinLimits:
                iteration = 0
                q = Q(env, self.environmentName)
                q.save()
                stateMapper.updateLimits(allHistoricObservations)
                stateMapper.save()

            self.runEpisode(env, q, stateMapper)
            self.learnFromPreviousExperience(q, stateMapper)
            

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

    def loadStepsData(self, stateMapper):
        if not os.path.isfile(self.stepsDataFileName()):
            self.stepsData = []
        else:
            with open(self.stepsDataFileName(), 'rb') as input:
                self.stepsData = pickle.load(input)
        for step in self.stepsData: 
            self.updateStepsIndex(step, stateMapper)


    def runEpisode(self, env, q, stateMapper):
        observation = env.reset()
        done = False
        stepCount = 0
        while not done:
            state = stateMapper.getState(observation)
            qValues = q.calculate(state)

            action = self.chooseActionBasedOnIndex(qValues,state)

            newObservation, reward, done, info = env.step(action)

            newState = stateMapper.getState(newObservation)
            q.learn(state, action, newState, reward, done)
            self.addStep({'observation':observation, 'action':action, 'newObservation':newObservation, 'reward':reward, 'done':done}, stateMapper)

            observation = newObservation
            env.render()
            stepCount += 1
            if stepCount % self.saveStepsAfter == 0:
                self.saveStepsData()
                print('Steps performed:'+str(stepCount)+'. Steps data saved.')
        
        self.saveStepsData()
        print('Steps performed:'+str(stepCount)+'(end of episode). Steps data saved.')
        self.logCounter(stepCount)
        

    def addStep(self, step, stateMapper):
        self.stepsData.append(step)
        self.updateStepsIndex(step, stateMapper)
    def updateStepsIndex(self, step, stateMapper):
        stateKey = str(stateMapper.getState(step['observation']))
        if not stateKey in self.stepsIndex.keys():
            self.stepsIndex[stateKey] = []
        self.stepsIndex[stateKey].append(step)


    def learnFromPreviousExperience(self, q, stateMapper):
        self.learnFromEndings(q, stateMapper)
        for _ in range(len(self.stepsIndex.keys()) * 1):
            indexKeys = list(self.stepsIndex.keys())
            randomKey = indexKeys[random.randint(0, len(indexKeys)-1)]
            stepsList = self.stepsIndex[randomKey]
            step = stepsList[random.randint(0, len(stepsList)-1)]
            self.learnStep(step, q, stateMapper)

    def learnStep(self, step, q, stateMapper):
        q.learn(stateMapper.getState(step['observation']), step['action'], stateMapper.getState(step['newObservation']), step['reward'], step['done'])

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

    def learnFromEndings(self, q, stateMapper):
        endings = self.findEndings()
        print('Endings:', len(endings))
        steps = [step for ending in endings for step in ending]
        for _ in range(10):
            for step in steps:
                self.learnStep(step, q, stateMapper)
            
        
