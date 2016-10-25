import os.path
import pickle

class Persister:
    def __init__(self, fileName):
        self.fileName = fileName
        
    def save(self, data):
        with open(self.fileName, 'wb') as output:
            pickle.dump(data, output, pickle.HIGHEST_PROTOCOL)

    def load(self, default):
        if not os.path.isfile(self.fileName):
            return default
        else:
            with open(self.fileName, 'rb') as input:
                return pickle.load(input)