import torch as pt
import numpy as np
from terminal import *

class Synapse:

    def __init__(self, pre, post, terms, delayRange):
        #weights = np.full(terms, 0.1)
        weights = np.random.uniform(0, 1.5, terms)
        delays = np.random.uniform(delayRange[0], delayRange[1], terms)
        delays = np.arange(delayRange[0], delayRange[1], (delayRange[1] - delayRange[0])/terms)
        self.terminals = self.createTerminals(delays, weights)
        self.pre = pre
        self.post = post

    def getPre(self):
        return self.pre

    def getPost(self):
        return self.post

    def getTerminals(self):
        return self.terminals

    def createTerminals(self, delays, weights):
        terminals = []
        for i, x in enumerate(delays):
            terminals.append(Terminal(x, weights[i]))
        return terminals
