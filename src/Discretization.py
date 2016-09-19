class Discretization:

    @staticmethod
    def create(observations):
        dimensions = len(observations[0])
        list = []
        for index in range(dimensions):
            list.append([observation[index]  for observation in observations])

        high = []
        low = []
        for dimension in range(dimensions):
            values = list[dimension]
            high.append(max(values))
            low.append(min(values))

        return Discretization(high, low)

    @staticmethod
    def createDefault(highValues, lowValues):
        return Discretization(highValues, lowValues)

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

    @staticmethod
    def equals(discretization1, discretization2):
        equal = (str(discretization1.high) == str(discretization2.high) and str(discretization1.low) == str(discretization2.low))
        if not equal :
            print("discretization not equal")
        return equal