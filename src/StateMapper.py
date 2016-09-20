import os.path
import pickle


class StateMapper:
    def __init__(self, high, low, environmentName):
        self.high = high
        self.low = low
        self.environmentName = environmentName
        self.ranges = 10
        self.extendRangePercent = 0.1

    def fileName(self):
        return self.environmentName + '.highs-lows.dat'

    def save(self):
        with open(self.fileName(), 'wb') as output:
            pickle.dump({'high': self.high, 'low': self.low}, output, pickle.HIGHEST_PROTOCOL)

    def extendVector(self):
        deltas = [h - l for h, l in zip(self.high, self.low)]
        deltas = [d * self.extendRangePercent for d in deltas]
        self.high = [h + d for h, d in zip(self.high, deltas)]
        self.low = [l - d for l, d in zip(self.low, deltas)]

    @staticmethod
    def load(dimensions, environmentName):
        stateMapper = StateMapper([], [], environmentName)
        if not os.path.isfile(stateMapper.fileName()):
            stateMapper.high = [0.0001] * dimensions
            stateMapper.low = [-0.0001] * dimensions
            return stateMapper
        with open(stateMapper.fileName(), 'rb') as input:
            config = pickle.load(input)
            stateMapper.high = config['high']
            stateMapper.low = config['low']
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
