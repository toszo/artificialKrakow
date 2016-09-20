import os.path
import pickle

class StateMapper:
    def __init__(self, high, low):
        self.high = high
        self.low = low
        self.ranges = 10

    fileName = "discretization.dat"
    def save(self):
        with open(StateMapper.fileName, 'wb') as output:
            pickle.dump({'high':self.high,'low':self.low}, output, pickle.HIGHEST_PROTOCOL)
     
    @staticmethod
    def load():
        if not os.path.isfile(StateMapper.fileName):
            raise Exception('discretization.dat not found')
        with open(StateMapper.fileName, 'rb') as input:
            config = pickle.load(input)
            return StateMapper(config['high'], config['low'])

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

    def update(self, observations):
        dimensions = len(observations[0])
        observationsPivot = []
        for index in range(dimensions):
            observationsPivot.append([observation[index] for observation in observations])

        high = []
        low = []
        for dimension in range(dimensions):
            values = observationsPivot[dimension]
            high.append(max(values))
            low.append(min(values))

        newHigh = [max(highs) for highs in list(zip(high, self.high))]
        newLow = [min(lows) for lows in list(zip(low, self.low))]
        changed = not (str(newHigh) == str(self.high) and str(newLow) == str(self.low))
        self.high = newHigh
        self.low = newLow
        return changed
