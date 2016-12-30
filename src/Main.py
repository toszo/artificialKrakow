import os.path
import pickle
import random
import matplotlib.pyplot as plt
import numpy as np

import gym

from StateMapper import StateMapper
from Q import Q

class Step:
    def __init__(self, observation, action, reward, nextObservation, done):
        self.observation = observation
        self.action = action
        self.reward = reward
        self.nextObservation = nextObservation
        self.done = done

class Main:
    def __init__(self, environmentName):
        self.steps = []
        self.environmentName = environmentName

    def execute(self):
        env = gym.make(self.environmentName)
        stateMapper = StateMapper.load(len(env.observation_space.high), self.environmentName)
        self.loadSteps()
        q = Q.load(env, stateMapper).convergeQ(self.steps)

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
                
                q = q.convergeQ(self.steps)                        
                if stepCount % 50 == 0:
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
            
            self.saveSteps()
            print('Steps performed:'+str(stepCount)+'(end of episode). Steps data saved.')
            

    def stepsFileName(self):
        return self.environmentName + '.steps.dat'
        
    def saveSteps(self):
        with open(self.stepsFileName(), 'wb') as output:
            pickle.dump(self.steps, output, pickle.HIGHEST_PROTOCOL)

    def loadSteps(self):
        if not os.path.isfile(self.stepsFileName()):
            self.steps = []
        else:
            with open(self.stepsFileName(), 'rb') as input:
                self.steps = pickle.load(input)
                print('Loaded ' + str(len(self.steps)) + ' steps.')

