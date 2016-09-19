class Q:
    def __init__(self, env):
        self.env = env

    learningRate = 1
    discountFactor = 0.9
    qMap = {}
    def calculate(self, state):
        defaultQ = 100
        result = [None]*self.env.action_space.n
        stateKey = str(state)
        if stateKey not in self.qMap.keys():
            self.qMap[stateKey] = dict()
        for action in range(self.env.action_space.n):
            if action not in self.qMap[stateKey].keys():
                self.qMap[stateKey][action] = defaultQ
            result[action] = self.qMap[stateKey][action]
        return result

   

    def learn(self, previousState, action, currentState, reward, done):
        oldValue = self.calculate(previousState)[action]
        learnedValue = self.getLearnedValue(currentState,reward) - oldValue
        self.qMap[str(previousState)][action] = oldValue + self.learningRate * learnedValue
        if done :
            for action in range(self.env.action_space.n):
                self.qMap[str(currentState)][action] = 0
        print('Q: ',self.qMap[str(previousState)][action])
        
    def getLearnedValue(self, currentState, reward):
        return reward + self.discountFactor * max(self.calculate(currentState))


