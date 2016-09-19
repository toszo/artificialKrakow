import gym
import random

class Q:
    learningRate = 1
    discountFactor = 0.2
    qMap = {}
    def calculate(self, state):
        defaultQ = 1
        result = [None]*env.action_space.n
        stateKey = str(state)
        if stateKey not in qMap.keys():
            qMap[stateKey] = dict()
        for action in range(env.action_space.n):
            if action not in qMap[stateKey].keys():
                qMap[stateKey][action] = defaultQ
            result[action] = qMap[stateKey][action]
        return result

    def learn(self,previousState, action, currentState, reward):
        oldValue = self.calculate(previousState)[action]
        result = oldValue + self.learningRate * (self.getLearnedValue(currentState,reward) - oldValue)
    def getLearnedValue(self, currentState, reward):
        return reward + self.discountFactor * max(self.calculate(currentState))

class MainClass:
    def getState(observation,env):
        DISCRETE = 5
        high = env.observation_space.high
        low = env.observation_space.low
        result = [None]*len(high)

        for i in range(len(high)):
            dx = (high[i]-low[i])/DISCRETE
            step = low[i]
            num = 0
            while(observation[i]>step):
                step+=dx
                num+=1
            result[i] = num
        return result

    def chooseAction(qValues):
        sum = sum(qValues)
        randValue = random.random() * sum
        index = 0
        for value in qValues:
            if randValue < value:
                return index
            randValue -= value
            index += 1   
      

    def execute():    
        env = gym.make('CartPole-v0')
    
        observation = env.reset()
        done = False

        while not done:
            state = getState(observation,env)
            qValues = calculateQs(state,env)
            action = chooseAction(qValues)
            previousState = state
            observation, reward, done, info = env.step(action)
            learnQ(previousState, action, state, reward)


def main(args=None):
    MainClass().execute()

if __name__ == "__main__":
    main()
