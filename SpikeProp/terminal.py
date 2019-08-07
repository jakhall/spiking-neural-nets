import numpy as np
import math

class Terminal:

    def __init__(self, delay, weight):
        self.delay = delay
        self.weight = weight

    def getDelay(self):
        return self.delay

    def setDelay(self, delay):
        self.delay = delay

    def getWeight(self):
        return self.weight

    def setWeight(self, weight):
        self.weight = weight

    def updateWeight(self, change):
        if self.weight + change < 0:
            self.weight = 0
        else:
            self.weight += change
