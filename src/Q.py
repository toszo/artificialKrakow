import os.path
import pickle

class StateStep:
    def __init__(self, state, action, reward, nextState, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.nextState = nextState
        self.done = done

class Q:
    def __init__(self, env, stateMapper, qDict = None):
        self.env = env
        self.stateMapper = stateMapper
        self.actions = range(self.env.action_space.n)
        self.qDict = {} if qDict is None else qDict
        self.discountFactor = 0.9999
        self.defaultQ = 0   

    def convergeQ(self, steps, policyNotChangedThreshold = 1):
        stateSteps = [StateStep(self.stateMapper.getState(step.observation), step.action, step.reward, self.stateMapper.getState(step.nextObservation), step.done) for step in steps]
        states = set([tuple(step.state) for step in stateSteps])
        stepsDict = {key: [] for key in [self.qDictKey(step.state, step.action) for step in stateSteps]}
        for step in stateSteps: 
            stepsDict[self.qDictKey(step.state, step.action)].append(step)

        q = self
        policyNotChangedCount = 0
        while True:            
            nextQ = q.B_Q(stepsDict)
            policy = q.policy(states)
            nextPolicy = nextQ.policy(states)
            q = nextQ
            if policy == nextPolicy:                
                policyNotChangedCount += 1
            else:               
                policyNotChangedCount = 0
            if policyNotChangedCount >= policyNotChangedThreshold:
                return nextQ

    def policy(self, states):
        policy = {}
        for state in states:
            qs = self.Qs(tuple(state))
            policy[tuple(state)] = qs.index(max(qs))
        return policy

    def B_Q(self, stepsDict):
        # Bellman Operator
        newQDict = {}
        for key in stepsDict.keys():
            steps = stepsDict[key]
            newQDict[key] = sum([(step.reward + self.discountFactor * max(self.Qs(step.nextState))) / len(steps) for step in steps])

        return Q(self.env, self.stateMapper, newQDict)

    def bestAction(self, state):
        policy = self.policy([state])
        return policy[tuple(state)]

    def Qs(self, state):
        keys = [self.qDictKey(state, action) for action in self.actions]
        return [self.qDict[key] if key in self.qDict.keys() else self.defaultQ for key in keys]

    def qDictKey(self, state, action):
        return (tuple(state), action)