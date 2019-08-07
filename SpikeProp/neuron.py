
import torch as pt
import numpy as np
import math
from synapse import *

class Neuron:
    global threshold

    def __init__(self, sign, preCons, terms, thresh, decay=5, pos=0, name="Neuron"):
        self.synapses = []
        self.sign = sign
        self.threshold = thresh
        self.default = thresh
        self.decay = decay
        self.spikeTimes = [0]
        self.hasSpiked = False
        self.name = name
        self.position = pos
        self.spikeHistory = []
        for p in preCons:
            self.synapses.append(Synapse(p, self, terms, [0, 100]))

    def LIF(self, time, sign):
        if time > 0:
            div = float(time) / self.decay
            return div * sign * math.exp(1 - div)
        else:
            return 0

    def LIFDer(self, time, sign):
        p = (-sign)*((float(time) - self.decay) * math.exp(1 - float(time)/self.decay))/self.decay**2
        return p

    def termPot(self, preSpikeTime, time, delay, preSign):
        time = float(time) - preSpikeTime - delay
        return self.LIF(time, preSign)

    def termPotDer(self, preSpikeTime, time, delay, preSign):
        #time = float(time) - preSpikeTime - delay
        time = 1
        return self.lifFunctionDer(time, preSign)

    def lifFunctionDer(self, time, nType):
        div = 1 - (float(time)/self.decay)
        p = div  * math.exp(div) / self.decay
        return p

    def updateISV(self, time, terms):
        ISV = 0
        if len(self.synapses) < 2:
            termPots = pt.Tensor(1, terms)
        else:
            termPots = pt.Tensor(len(self.synapses), terms)
        for x, s in enumerate(self.synapses):
            terminals = s.getTerminals()
            pre = s.getPre()
            spikeTime = pre.getLastSpike()
            termPots[x] = pt.Tensor(np.zeros(terms))
            if time >= spikeTime and spikeTime > 0:
                for i, t in enumerate(terminals):
                    pot = self.termPot(spikeTime, time, t.getDelay(), pre.getSign())
                    termPots[x, i] = t.getWeight() * pot
                    ISV += t.getWeight() * pot
                    if ISV >= self.threshold:
                        self.generateSpike(time)
                        #print(self.name, ":", ISV, self.threshold, time)
        return ISV, termPots


    def generateSpike(self, time):
        if not self.hasSpiked:
            self.hasSpiked = True
            self.spikeTimes.append(time)
            #print(self.name, " Spiked At: ", time, self.threshold)


    def getPrevSpike(self, pos):
        return self.spikeHistory[pos]

    def getHistory(self):
        return self.spikeHistory

    def saveLastSpike(self):
        self.spikeHistory.append(self.spikeTimes[-1])

    def resetSpikeHist(self):
        self.spikeHistory = []


    def getSign(self):
        return self.sign

    def getSpikes(self):
        return self.spikeTimes

    def getLastSpike(self):
        return self.spikeTimes[-1]

    def resetSpikes(self):
        self.spikeTimes = [0]
        self.hasSpiked = False

    def resetThreshold(self):
        self.threshold = self.default

    def setSynapses(self, preCons):
        self.synapses = preCons

    def getSynapses(self):
        return self.synapses

    def getThreshold(self):
        return self.threshold

    def setThreshold(self, thresh):
        self.threshold = thresh

    def hasNeuronSpiked(self):
        return self.hasSpiked

    def getPosition(self):
        return self.position

    def getName(self):
        return self.name

    def getPreCons(self):
        x = []
        [x.append(s.getPre()) for s in self.synapses]
        return x

    def getAverageWeight(self):
        weight = 0
        for s in self.synapses:
            for t in s.getTerminals():
                weight += t.getWeight()/len(s.getTerminals())
        return weight


    def getSumWeights(self):
        sum = 0
        for s in self.synapses:
            for t in s.getTerminals():
                sum += t.getWeight()
        return sum
