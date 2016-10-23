import os.path
import pickle
import random
import gym
import numpy
from TensorFlowAnn import TensorFlowAnn

class MainAnn:
    def __init__(self, environmentName):
        self.environmentName = environmentName
       
    def chooseRandomAction(self, Qs):
        return random.randint(0, len(Qs) - 1)

    def chooseBestAction(self, Qs):
        qs = list(numpy.copy(Qs).flatten())
        return qs.index(max(qs))

    def chooseAction(self,Qs):
        if random.randint(0,100) > 90:
            return self.chooseRandomAction(Qs)
        else:
            return self.chooseBestAction(Qs)

    def execute(self):
        env = gym.make(self.environmentName)
        ann = TensorFlowAnn(self.environmentName, [len(env.observation_space.high), 20, env.action_space.n])
        while True:
            stepsData = []
            for _ in range(0, 1):
                newSteps = self.runEpisode(env, ann)
                stepsData += newSteps
            self.trainAnn(ann, stepsData)

    def trainAnn(self, ann, stepsData):
        print('training ANN')
        for _ in range(0, 10):
            print('calculate train samples')
            (observations, Qs) = self.prepareTrainSamples(ann, stepsData)
            ann.train(observations, Qs)
        print('training completed')

    def prepareTrainSamples(self, ann, stepsData):
        learningRate = 0.1
        discountFactor = 0.9
        observations = [step['observation'] for step in stepsData]
        Qs = [ann.calculate(observation) for observation in observations]
        for Q, step in zip(Qs, stepsData):
            newValue = discountFactor * max(ann.calculate(step['newObservation'])) + step['reward']
            oldValue = Q[step['action']] 
            Q[step['action']] = oldValue + learningRate * (newValue - oldValue)
        return (observations, Qs)

    def runEpisode(self, env, ann):
        print('starting episode')
        done = False
        stepsData = []
        observation = env.reset()
        while not done:
            Qs = ann.calculate(observation)
            action = self.chooseAction(Qs)
            newObservation, reward, done, info = env.step(action)
            stepsData.append({'observation':observation, 'action':action, 'newObservation':newObservation, 'reward':reward, 'done':done})
            env.render()
            observation = newObservation
        print('episode completed')
        return stepsData
