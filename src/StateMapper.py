import os.path
import pickle
import numpy

class StateMapper:
    def __init__(self, environmentName):
        self.high = []
        self.low = []        
        self.highFromData = []
        self.lowFromData = []
        self.environmentName = environmentName
        self.ranges = 10
        self.extendRangePercent = 0.1

    @staticmethod
    def fileName(environmentName):
        return environmentName + '.highs-lows.dat'

    def save(self):
        with open(StateMapper.fileName(self.environmentName), 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        
        dimensions = numpy.array([self.high, self.highFromData, self.low, self.lowFromData]).transpose()
        print('New StateMapper configuration saved')
        print('high (from data), low (from data)')
        for dimension in dimensions:
            print(str(dimension[0]) + ' (' + ('Y' if dimension[1] else 'N') + '), ' + str(dimension[2]) + ' (' + ('Y' if dimension[3] else 'N') + ')')

    @staticmethod
    def load(dimensions, environmentName):
        if not os.path.isfile(StateMapper.fileName(environmentName)):
            stateMapper = StateMapper(environmentName)
            stateMapper.high = [0.0001] * dimensions
            stateMapper.low = [-0.0001] * dimensions
            stateMapper.highFromData = [False] * dimensions
            stateMapper.lowFromData = [False] * dimensions
        else:
            with open(StateMapper.fileName(environmentName), 'rb') as input:
                stateMapper = pickle.load(input)

        dimensions = numpy.array([stateMapper.high, stateMapper.highFromData, stateMapper.low, stateMapper.lowFromData]).transpose()
        print('StateMapper configuration loaded')
        print('high (from data), low (from data)')
        for dimension in dimensions:
            print(str(dimension[0]) + ' (' + ('Y' if dimension[1] else 'N') + '), ' + str(dimension[2]) + ' (' + ('Y' if dimension[3] else 'N') + ')')
        
        return stateMapper

    def getState(self, observation):
        result = [None] * len(self.high)

        for i in range(len(self.high)):
            dx = (self.high[i] - self.low[i]) / self.ranges
            step = self.low[i]

            value = int((observation[i] - self.low[i]) / dx)
            if value < 0:
                value = 0
            elif value >= self.ranges:
                value = self.ranges - 1

            result[i] = value
        return result

    def observationsWithinLimits(self, observations):
        observations_high = numpy.amax(observations, axis=0)
        observations_low = numpy.amin(observations, axis=0)

        withinHighLimit = [oh <= h for h, oh in zip(self.high, observations_high)]
        withinLowLimit  = [l <= ol for l, ol in zip(self.low, observations_low)]

        return all(withinHighLimit) and all(withinLowLimit)

    def updateLimits(self, observations):
        observations_high = numpy.amax(observations, axis=0)
        observations_low = numpy.amin(observations, axis=0)

        for index in range(len(self.high)):
            extendHigh = False
            if not self.highFromData[index] or self.high[index] < observations_high[index]:
                self.high[index] = observations_high[index]
                self.highFromData[index] = True
                extendHigh = True
            extendLow = False
            if not self.lowFromData[index] or observations_low[index] < self.low[index]:
                self.low[index] = observations_low[index]
                self.lowFromData[index] = True
                extendLow = True

            high = self.high[index] 
            low = self.low[index]
            delta = (1 + self.extendRangePercent)*(high - low)
            if extendHigh:
                self.high[index] = low + delta
            if extendLow:
                self.low[index] = high - delta

