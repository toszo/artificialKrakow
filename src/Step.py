class Step:
    def __init__(self, observation, action, reward, nextObservation, done):
        self.observation = observation
        self.action = action
        self.reward = reward
        self.nextObservation = nextObservation
        self.done = done