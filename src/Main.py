import random
#import cPickle
import gym

from Discretization import Discretization
from Q import Q


class MainClass:
    def __init__(self):        
        self.episodeData = []

    def chooseRandomAction(self, qValues):
        randValue = random.random() * sum(qValues)
        index = 0
        for value in qValues:
            if randValue < value:
                return index
            randValue -= value
            index += 1

    def chooseBestAction(self, qValues):
        return qValues.index(max(qValues))


    def execute(self):
        env = gym.make('CartPole-v0')
        q = Q.load(env)
        discreter = Discretization.load()

        iterate = 0
        while True:
            observation = env.reset()
            self.runEpisode(env, observation, q, discreter)
            self.learnFromPreviousExperience(q, discreter)
            allHistoricObservations = [episode[0] for episode in self.episodeData]
            changed = discreter.update(allHistoricObservations)
            if changed:
                q = Q(env)
                discreter.save()   
                print('new discretization.dat saved')       
            iterate+=1  
            if iterate % 10 == 0:
                q.save()        

    # def saveEpisodeData(self):
    #     file = open("episodeData.dat", "w")
    #     file.write(cPickle.dumps(self.episodeData))

    # def loadEpisodeData(self):
    #     self.episodeData = cPickle.load(open("episodeData.dat","rb"))

    def runEpisode(self, env, observation, q, discreteConverter):
        done = False
        stepCounter = 0
        while not done:
            state = discreteConverter.getState(observation)
            qValues = q.calculate(state)
            action = self.chooseRandomAction(qValues)

            newObservation, reward, done, info = env.step(action)
           
            newState = discreteConverter.getState(newObservation)
            q.learn(state, action, newState, reward, done)           
            self.episodeData.append([observation, action, newObservation, reward,done])
            #env.render()
            stepCounter += 1
        

    def learnFromPreviousExperience(self, q, discreter):
        for example in self.episodeData:
            q.learn(discreter.getState(example[0]),example[1],discreter.getState(example[2]),example[3],example[4])


def main(args=None):
    MainClass().execute()

if __name__ == "__main__":
    main()