import os.path
import pickle

class Discretization:
    fileName = "discretization.dat"
    def save(self):
        with open(Discretization.fileName, 'wb') as output:
            pickle.dump({'high': self.high, 'low':self.low}, output, pickle.HIGHEST_PROTOCOL)
     
    @staticmethod
    def load():
        if not os.path.isfile(Discretization.fileName):
            raise Exception('Discretization file not found')
        with open(Discretization.fileName, 'rb') as input:
            config = pickle.load(input)
            return Discretization(config['high'], config['low'])

    def __init__(self, high, low):
        self.high = high
        self.low = low

    def getState(self, observation):
        RANGES = 10
        result = [None] * len(self.high)

        for i in range(len(self.high)):
            if observation[i] < self.low[i]:
                result[i] =0
                continue
            if observation[i] >= self.high[i]:
                result[i] = len(self.high) - 1
                continue

            dx = (self.high[i] - self.low[i]) / RANGES
            step = self.low[i]

            num = 0
            while (observation[i] >= step):
                step += dx
                num += 1
            result[i] = num
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
