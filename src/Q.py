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
        self.discountFactor = 0.9
        self.defaultQ = 0

    def fileName(env):
        return env.spec.id + '.q.dat'
    def save(self):
        with open(Q.fileName(self.env), 'wb') as output:
            pickle.dump(self.qDict, output, pickle.HIGHEST_PROTOCOL)
    @staticmethod
    def load(env, stateMapper):
        if not os.path.isfile(Q.fileName(env)):
            return Q(env, stateMapper, {})
        with open(Q.fileName(env), 'rb') as input:
            return Q(env, stateMapper, pickle.load(input))

    def convergeQ(self, steps):
        stateSteps = [StateStep(self.stateMapper.getState(step.observation), step.action, step.reward, self.stateMapper.getState(step.nextObservation), step.done) for step in steps]
        states = set([tuple(step.state) for step in stateSteps])
        stepsDict = {key: [] for key in [self.qDictKey(step.state, step.action) for step in stateSteps]}
        for step in stateSteps: 
            stepsDict[self.qDictKey(step.state, step.action)].append(step)

        q = self
        count = 0
        while True:
            count += 1
            nextQ = q.B_Q(stepsDict)
            policy = q.policy(states)
            nextPolicy = nextQ.policy(states)
            if policy == nextPolicy:                
                print('Converged in ' + str(count) + ' iterations.')
                return nextQ
            else:               
                q = nextQ

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
            newQDict[key] = sum([step.reward + self.discountFactor / len(steps) * max(self.Qs(step.nextState)) for step in steps])

        return Q(self.env, self.stateMapper, newQDict)

    def bestAction(self, state):
        Qs = self.Qs(state)
        return Qs.index(max(Qs))

    def Qs(self, state):
        keys = [self.qDictKey(state, action) for action in self.actions]
        return [self.qDict[key] if key in self.qDict.keys() else self.defaultQ for key in keys]

    def qDictKey(self, state, action):
        return (tuple(state), action)