import os.path
import pickle
import random
import matplotlib.pyplot as plt
import numpy as np

import gym

from StateMapper import StateMapper
from Q import Q
from Persister import Persister
from Step import Step

class Main:
    def __init__(self, environmentName):
        self.steps = []
        self.stepsPersister = Persister(environmentName + '.steps.dat')
        self.environmentName = environmentName      

    def execute(self):
        env = gym.make(self.environmentName)
        self.steps = self.stepsPersister.load([])
        stateMapper = StateMapper.load(len(env.observation_space.high), self.environmentName)
        stateMapper.updateLimits([step.observation for step in self.steps])
        q = Q(env, stateMapper).convergeQ(self.steps, 40)

        while True:
            observation = env.reset()
            done = False
            stepCount = 0
            while not done:
                state = stateMapper.getState(observation)
                action = q.bestAction(state)
                nextObservation, reward, done, info = env.step(action)
                step = Step(observation, action, reward, nextObservation, done)
                self.steps.append(step)

                observation = nextObservation
                env.render()

                observations = [step.observation for step in self.steps]
                if not stateMapper.observationsWithinLimits(observations):
                    stateMapper.updateLimits(observations)
                    q = Q(env, stateMapper)
                    print('Recreated Q')
                
                if stepCount % 50 == 0:
                    q = q.convergeQ(self.steps)                     
                    states = [stateMapper.getState(step.observation) for step in self.steps]
                    policy = q.policy(states)

                    values = np.zeros((stateMapper.ranges, stateMapper.ranges))
                    for state in policy.keys():
                        values[state[1], state[0]] = max(q.Qs(state))
                    policies = np.zeros((stateMapper.ranges, stateMapper.ranges))
                    for state in policy.keys():
                        policies[state[1], state[0]] = policy[state] + 1

                    plt.ion()
                    plt.figure(0)
                    plt.imshow(values, cmap='hot', interpolation='nearest')
                    plt.figure(1)
                    plt.imshow(policies, cmap='hot', interpolation='nearest')
                    plt.pause(0.001)
                stepCount += 1
            
            self.stepsPersister.save(self.steps)
            print('Steps performed:'+str(stepCount)+'(end of episode). Steps data saved.')
  
