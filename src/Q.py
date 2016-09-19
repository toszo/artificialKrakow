class Q:
    learningRate = 1
    discountFactor = 0.2
    qMap = {}
    def calculate(self, state):
        return 'lol'
    def learn(self,previousState, action, currentState, reward):
        oldValue = self.calculate(previousState)[action]
        result = oldValue + self.learningRate * (self.getLearnedValue(currentState,reward) - oldValue)
    def getLearnedValue(self, currentState, reward):
        return reward + self.discountFactor * max(self.calculate(currentState))
