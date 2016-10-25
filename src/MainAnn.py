import os.path
import pickle
import random
import gym
import numpy
import time
import math
from TensorFlowAnn import TensorFlowAnn
from Persister import Persister

class MainAnn:
    def __init__(self, environmentName):
        self.stepsDataPersister = Persister(environmentName + '.stepsData.dat')
        self.env = gym.make(environmentName)
        self.ann = TensorFlowAnn(environmentName, [len(self.env.observation_space.high), 20, self.env.action_space.n])

       
    def chooseRandomAction(self, Qs):
        return random.randint(0, len(Qs) - 1)

    def chooseBestAction(self, Qs):
        action = list(Qs).index(max(Qs))
        return action

    def chooseAction(self,Qs):
        if random.randint(0,100) > 100:
            return self.chooseRandomAction(Qs)
        else:
            return self.chooseBestAction(Qs)
        # return self.chooseBestAction(Qs)

    def execute(self):
        stepsData = self.stepsDataPersister.load([])
        while True:
            self.trainAnn(self.ann, stepsData)
            for i in range(0, 10):
                print('episode', i)
                newSteps = self.runEpisode(self.env, self.ann)
                stepsData += newSteps
            self.stepsDataPersister.save(stepsData)
    
    def train(self):
        stepsData = self.stepsDataPersister.load([])
        self.trainAnn(self.ann, stepsData)

    def trainAnn(self, ann, stepsData):
        print('training ann')
        for i in range(0, 10):
            print(i, 'calculate train samples:', len(stepsData))
            (observations, Qs) = self.prepareTrainSamples(ann, stepsData)
            print('training observations->Qs')
            ann.train(observations, Qs)
        print('training completed')    

    def reservoir_sampling(self, iterator, K):
        result = []
        N = 0
        for item in iterator:
            N += 1
            if len( result ) < K:
                result.append( item )
            else:
                s = int(random.random() * N)
                if s < K:
                    result[ s ] = item
        return result

    def prepareTrainSamples(self, ann, stepsData):
        learningRate = 0.3
        discountFactor = 0.9
        batch = self.reservoir_sampling(stepsData, 10000)
        observations = [step['observation'] for step in batch]
        Qs = ann.calculateBatch(observations)
        diffs = []
        for Q, step in zip(Qs, batch):
            if step['done']:
                newValue = step['reward']
            else:
                NextQs = ann.calculate(step['newObservation'])
                newValue = discountFactor * max(NextQs) + step['reward']
            oldValue = Q[step['action']]
            diffs.append(abs(newValue - oldValue)) 
            Q[step['action']] = oldValue + learningRate * (newValue - oldValue)
        print('mean diffs', numpy.array(diffs).mean())
        return (observations, Qs)

    def runEpisode(self, env, ann):
        done = False
        stepsData = []
        observation = env.reset()
        rewardSum = 0
        while not done:
            Qs = ann.calculate(observation)
            action = self.chooseAction(Qs)
            newObservation, reward, done, info = env.step(action)
            stepsData.append({'observation':observation, 'action':action, 'newObservation':newObservation, 'reward':reward, 'done':done})
            rewardSum += reward
            print('reward', reward, rewardSum)
            env.render()
            observation = newObservation
        return stepsData
