import random
import cPickle
import gym

from Discretization import Discretization
from Q import Q


class MainClass:
    def __init__(self):        
        self.episodeData = []
    def chooseAction(self, qValues):
        randValue = random.random() * sum(qValues)
        index = 0
        for value in qValues:
            if randValue < value:
                return index
            randValue -= value
            index += 1

    def execute(self):
        env = gym.make('CartPole-v0')
        q = Q(env)
        discreter = Discretization.createDefault(env.observation_space.high, env.observation_space.low)

        self.loadEpisodeData()
        iterate = 0
        while True:
            observation = env.reset()
            self.runEpisode(env, observation, q, discreter)
            self.learnFromPreviousExperience(q)
            newDiscreter = Discretization.create([ed[0] for ed in self.episodeData])
            if not Discretization.equals(newDiscreter,discreter):
                q = Q(env)
                discreter = newDiscreter
            print('new episode')
            iterate+=1
            if iterate % 100 == 0:
               self.saveEpisdeData()

    def saveEpisodeData(self):
        file = open("episodeData.dat", "w")
        file.write(cPickle.dumps(self.episodeData))

    def loadEpisodeData(self):
        self.episodeData = cPickle.load(open("episodeData.dat","rb"))

    def runEpisode(self, env, observation, q, discreteConverter):
        done = False
        stepCounter = 0
        while not done:
            state = discreteConverter.getState(observation)
            qValues = q.calculate(state)
            action = self.chooseAction(qValues)

            newObservation, reward, done, info = env.step(action)

            newState = discreteConverter.getState(newObservation)
            q.learn(state, action, newState, reward)           
            self.episodeData.append([observation, action, newObservation, reward])
            env.render()
            stepCounter += 1
        print(stepCounter)

    def learnFromPreviousExperience(self, q):
        for example in self.episodeData:
            q.learn(example[0],example[1],example[2],example[3])


def main(args=None):
    MainClass().execute()

if __name__ == "__main__":
    main()