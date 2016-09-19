import random
import os.path
import pickle
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
            if randValue <= value:
                return index
            randValue -= value
            index += 1

    def chooseBestAction(self, qValues):
        return qValues.index(max(qValues))


    def execute(self):
        env = gym.make('CartPole-v0')
        q = Q.load(env)
        discreter = Discretization.load()
        self.load()
        self.clearCouterLog()
        iterate = 0
        while True:
            observation = env.reset()
            reachedStepNumber = self.runEpisode(env, observation, q, discreter)
            self.logCounter(iterate,reachedStepNumber)
            self.learnFromPreviousExperience(q, discreter)
            allHistoricObservations = [episode['observation'] for episode in self.episodeData]
            changed = discreter.update(allHistoricObservations)
            if changed:
                q = Q(env)
                q.save()
                discreter.save()   
                print('new discretization.dat saved')       
            iterate+=1  
            if iterate % 1 == 0:
                q.save() 
                self.save()      
                print('saved q.dat and episodeData.dat') 

    fileName = 'episodeData.dat'   
    counterFileName = 'counter.dat'

    def clearCouterLog(self):
        file = open(self.counterFileName,"w")
        file.seek(0)
        file.truncate()

    def logCounter(self,iterate,number):
        file = open(self.counterFileName,"a")
        file.write(str(iterate)+" "+str(number)+"\n")

    def save(self):
        with open(self.fileName, 'wb') as output:
            pickle.dump(self.episodeData, output, pickle.HIGHEST_PROTOCOL)

    def load(self):
        if not os.path.isfile(MainClass.fileName):
            self.episodeData = []
        else:
            with open(self.fileName, 'rb') as input:
                self.episodeData = pickle.load(input)

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
            self.episodeData.append({'observation':observation, 'action':action, 'newObservation':newObservation, 'reward':reward, 'done':done})
            
            #nowa obserwacja staje sie biezaca
            observation = newObservation
            #env.render()
            stepCounter += 1
        return stepCounter

    def learnFromPreviousExperience(self, q, discreter):
        for _ in range(len(self.episodeData) * 10):
            index = random.randint(0, len(self.episodeData)-1)
            episode = self.episodeData[index]
            q.learn(discreter.getState(episode['observation']), episode['action'], discreter.getState(episode['newObservation']), episode['reward'], episode['done'])


def main(args=None):
    MainClass().execute()

if __name__ == "__main__":
    main()