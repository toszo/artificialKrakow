import os.path
import pickle

class Q:
    fileName = "q.dat"
    def __init__(self, env):
        self.env = env
        self.qMap = {}
        self.learningRate = 0.2
        self.discountFactor = 0.9
        self.defaultQ = 1

    def save(self):
        with open(Q.fileName, 'wb') as output:
            pickle.dump(self.qMap, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(env):
        if not os.path.isfile(Q.fileName):
            return Q(env)
        with open(Q.fileName, 'rb') as input:
            qMap = pickle.load(input)
            q = Q(env)
            q.qMap = qMap
            return q

    def calculate(self, state):
        result = [None]*self.env.action_space.n
        stateKey = str(state)
        if stateKey not in self.qMap.keys():
            self.qMap[stateKey] = dict()
        for action in range(self.env.action_space.n):
            if action not in self.qMap[stateKey].keys():
                self.qMap[stateKey][action] = self.defaultQ
            result[action] = self.qMap[stateKey][action]
        return result

    def learn(self, previousState, action, currentState, reward, done):
        oldValue = self.calculate(previousState)[action]
        learnedValue = self.getLearnedValue(currentState, reward)
        newValue = oldValue + self.learningRate * (learnedValue - oldValue)
        #if oldValue != learnedValue:
            #print('Old: ' + str(oldValue) + ', New: ' + str(newValue))
        self.qMap[str(previousState)][action] = newValue
        if done:
            for action in range(self.env.action_space.n):
                self.qMap[str(currentState)][action] = 0
        
    def getLearnedValue(self, currentState, reward):
        return reward + self.discountFactor * max(self.calculate(currentState))

    