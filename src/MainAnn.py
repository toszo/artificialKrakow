import os.path
import pickle
import random
import gym
import numpy as np
import time
import math
from TensorFlowAnn import TensorFlowAnn
from Persister import Persister

class MainAnn:
    def __init__(self, environmentName):
        self.stepsDataPersister = Persister(environmentName + '.stepsData.dat')
        self.env = gym.make(environmentName)
        self.actions = range(0, self.env.action_space.n)
        self.ann = TensorFlowAnn(environmentName, len(self.actions), [len(self.env.observation_space.high), 20])

       
    def chooseRandomAction(self, Qs):
        return random.randint(0, len(Qs) - 1)

    def chooseBestAction(self, Qs):
        action = list(Qs).index(max(Qs))
        return action

    def chooseAction(self,Qs):
        if random.randint(0,100) > 80:
            return self.chooseRandomAction(Qs)
        else:
            return self.chooseBestAction(Qs)

    def execute(self):
        stepsData = self.stepsDataPersister.load([])
        while True:
            newSteps = self.runEpisodes(self.env, self.ann)
            stepsData += newSteps
            self.stepsDataPersister.save(stepsData)

            self.trainSs(self.ann, stepsData)
            self.trainRs(self.ann, stepsData)
            self.trainQs(self.ann, stepsData)
            self.ann.saveNetworks()
    
    def trainQs(self, ann, stepsData):
        for i in range(0, 10):
            for action in self.actions:       
                print(i, 'training Qs, action', action)
                stepsForAction = [step for step in stepsData if step['action'] == action]
                print('  preparing Qs train samples', len(stepsForAction))
                (observations, Qs) = self.prepareQsTrainSamples(ann, action, stepsForAction)
                print('  training Qs')
                ann.trainQs(action, observations, Qs)

    def prepareQsTrainSamples(self, ann, action, stepsForAction):
        learningRate = 0.5
        self.discountFactor = 0.99
        observations = [step['observation'] for step in stepsForAction]
        newObservations = [step['newObservation'] for step in stepsForAction]
        Qs = ann.calculateBatch(ann.Qs, observations)
        Rs = ann.calculateBatch(ann.Rs, observations)
        Ss = ann.calculateBatch(ann.Ss, observations)
        QsFromSs = self.QsFromSs(ann, Ss)     
        nextQs = ann.calculateBatch(ann.Qs, newObservations)
        nextRs = ann.calculateBatch(ann.Rs, newObservations)
        nextSs = ann.calculateBatch(ann.Ss, newObservations)
        nextQsFromSs = self.QsFromSs(ann, nextSs)
        outQs = []
        for Q, R, S, QFromS, nextQ, nextR, nextS, nextQFromS, step in zip(Qs, Rs, Ss, QsFromSs, nextQs, nextRs, nextSs, nextQsFromSs, stepsForAction):
            if step['done']:
                newValue = step['reward']
            else:
                F = self.discountFactor * self.Phi(nextQ, nextR, nextS, nextQFromS) - self.Phi(Q, R, S, QFromS)
                newValue = self.discountFactor * max(nextQ)[0] + step['reward'] + F
            oldValue = Q[action][0]
            outQs.append([oldValue + learningRate * (newValue - oldValue)])
        return (observations, outQs)

    def Phi(self, Q, R, S, QFromS):
        sum = 0
        for action in self.actions:
            aQ = R[action][0] + self.discountFactor * max(QFromS[action])[0]
            sum += (Q[action][0] - aQ)**2
        return sum / len(self.actions)

    def QsFromSs(self, ann, Ss):
        reshapedSs = np.copy(Ss).reshape([Ss.shape[0]*Ss.shape[1], Ss.shape[2]])
        Qs = ann.calculateBatch(ann.Qs, reshapedSs)
        return Qs.reshape([Ss.shape[0], Ss.shape[1], Ss.shape[1], 1])


    def trainSs(self, ann, stepsData):
        for action in self.actions:       
            print('training Ss, action', action)
            stepsForAction = [step for step in stepsData if step['action'] == action]
            print('  preparing Ss train samples', len(stepsForAction))
            (observations, Ss) = self.prepareSsTrainSamples(ann, action, stepsForAction)
            print('  training Ss')
            ann.trainSs(action, observations, Ss)

    def prepareSsTrainSamples(self, ann, action, stepsForAction):
        observations = [step['observation'] for step in stepsForAction]
        Ss = [step['newObservation'] for step in stepsForAction]
        return (observations, Ss)


    def trainRs(self, ann, stepsData):
        for action in self.actions:       
            print('training Rs, action', action)
            stepsForAction = [step for step in stepsData if step['action'] == action]
            print('  preparing Rs train samples', len(stepsForAction))
            (observations, Rs) = self.prepareRsTrainSamples(ann, action, stepsForAction)
            print('  training Rs')
            ann.trainRs(action, observations, Rs)

    def prepareRsTrainSamples(self, ann, action, stepsForAction):
        observations = [step['observation'] for step in stepsForAction]
        Rs = [[step['reward']] for step in stepsForAction]
        return (observations, Rs)


    def runEpisodes(self, env, ann):
        stepsData = []
        for i in range(0, 10):
            print('episode', i)
            done = False
            observation = env.reset()
            rewardSum = 0
            newSteps = []
            while not done:
                Qs = ann.calculateBatch(ann.Qs, [observation]).reshape([len(self.actions)])
                action = self.chooseAction(Qs)
                newObservation, reward, done, info = env.step(action)
                newSteps.append({'observation':observation, 'action':action, 'newObservation':newObservation, 'reward':reward, 'done':done})
                rewardSum += reward
                #print('reward: {0: 7.2f}, {1: 7.2f}'.format(reward, rewardSum))
                env.render()
                observation = newObservation
            stepsData += newSteps
        return stepsData
