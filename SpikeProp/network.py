import math
import numpy as np
from neuron import *
import torch as pt
import torch.nn.functional as F

class Network:
    def __init__(self, layers, decay, threshold, terminals):
        self.decay = decay
        self.threshold = threshold
        self.terminals = terminals
        self.neuronNo  = np.sum(layers)
        self.input = []
        self.hidden = []
        self.output = []
        self.layers = []
        self.Failed = False

        layer = []
        sign = 1
        for n in range(0, layers[0]):
            self.input.append(Neuron(sign, [], self.terminals, self.threshold, self.decay, n, "input_" + str(n)))
        for l in range(1, len(layers) - 1):
            for n in range(0, layers[l]):
                if n < 1:
                    sign = -1 # -1
                else:
                    sign = 1
                if l > 1:
                    layer.append(Neuron(sign, self.hidden[n-1], self.terminals, self.threshold, self.decay, n, "hidden_" + str(n)))
                else:
                    layer.append(Neuron(sign, self.input, self.terminals, self.threshold, self.decay, n,"hidden_" + str(n)))
            self.hidden.append(layer)
        for n in range(0, layers[2]):
            self.output.append(Neuron(sign, self.hidden[-1], self.terminals, self.threshold, self.decay, n, "output_" + str(n)))
        self.layers.append(self.input)
        for l in self.hidden:
            self.layers.append(l)
        self.layers.append(self.output)


    def train(self, data, learnRate, batchSize=1, sgd=1, trace=False):
        data = np.array(data)
        sample_index = np.random.choice(len(data), int(len(data)*sgd), replace=False)
        sampled_data = data[sample_index]
        batch_data = np.array_split(sampled_data, len(data)/batchSize)
        for batch in batch_data:
            self.resetAllHistory()
            for spikes in batch:
                predSpikes, isvHistory, termHistory, attempts = self.ForwardProp(spikes[0])
            changes, delta = self.BackProp(self.layers[-1], batch[:, 1], learnRate)
            self.updateLayerWeights(self.layers[-1], changes)
            for l in range(len(self.layers) - 2, 0, -1):
                changes, delta = self.BackProp(self.layers[l], self.layers[l + 1], learnRate, delta)
                self.updateLayerWeights(self.layers[l], changes)
        if trace:
            mse = []
            predicted_spikes = []
            for spikes in data:
                predSpikes, isvHistory, termHistory, attempts = self.ForwardProp(spikes[0], trace)
                mse.append(self.MSE(spikes[1], predSpikes))
                predicted_spikes.append(predSpikes)
            print("Predicted:", predicted_spikes, "MSE:", mse, "Attempts:", attempts)
        return isvHistory, termHistory

    def train2(self, data, learnRate, trace=False):
        input_spikes = [x[0] for x in data]
        target_spikes = [x[1] for x in data]
        isvHistory = []
        termHistory = []
        mse = []
        predicted_spikes = []
        spike_list = []
        for g, target in enumerate(target_spikes):
            predSpikes, spikeList, isvHistory, termHistory, attempts = self.ForwardProp(input_spikes[g])
            predicted_spikes.append(predSpikes)
            spike_list.append(spikeList)
        changes, delta = self.BackProp2(self.layers[-1], spike_list, target_spikes, learnRate)
        self.updateLayerWeights(self.layers[-1], changes)
        for l in range(len(self.layers) - 2, 0, -1):
            changes, delta = self.BackProp2(self.layers[l], spike_list, self.layers[l + 1], learnRate, delta, l)
            self.updateLayerWeights(self.layers[l], changes)

        mse = []
        predicted_spikes = []
        if trace:
            for t, input in enumerate(input_spikes):
                predSpikes, spikeList, isvHistory, termHistory, attempts = self.ForwardProp(input, trace)
                mse.append(self.MSE(target_spikes[t], predSpikes))
                predicted_spikes.append(predSpikes)
            print(target, "Predicted:", predicted_spikes, "MSE:", mse, "Attempts:", attempts)
        return isvHistory, termHistory


    def ForwardProp(self, input, trace=False):
            attempts = 0
            predicted_spikes = []
            allFired = False
            maxSynapses = 5
            isvHistory = pt.Tensor(200, self.neuronNo)
            termHistory = pt.Tensor(200, self.neuronNo, maxSynapses, self.terminals)
            self.resetAllThresholds()
            while attempts < 100 and not allFired:
                self.resetAllSpikes()
                timestep = 0
                for i, n in enumerate(self.input[:-1]):
                #for i, n in enumerate(self.input):
                    n.generateSpike(input[i])
                self.input[-1].generateSpike(1)
                while(timestep < 200 and not self.forwardComplete(trace)):
                    currentN = 0
                    for layer in self.layers:
                        for neuron in layer:
                            isv, term = neuron.updateISV(timestep, self.terminals)
                            isvHistory[timestep, currentN] = pt.Tensor([isv])
                            term = F.pad(term, [0, 0, 0, maxSynapses - term.shape[0]])
                            termHistory[timestep, currentN] = term
                            currentN += 1
                    timestep+=1
                if self.checkSpiked():
                    predicted_spikes.append(self.output[0].getLastSpike())
                    allFired = True
                else:
                    attempts+=1
            if not allFired:
                print("Failed to Fire")
            self.recordSpikes()
            return predicted_spikes, isvHistory, termHistory, attempts

    def forwardComplete(self, trace):
        if trace:
            return False
        spiked = True
        for n in self.output:
            if not n.hasNeuronSpiked():
                spiked = False
        return spiked

    def recordSpikes(self):
        spikeList = []
        for layer in self.layers:
            [n.saveLastSpike() for n in layer]

    def failCheck(self):
        print(self.Failed)
        return self.Failed

    def checkSpiked(self):
        spiked = True
        for layer in self.layers:
            for neuron in layer:
                if not neuron.hasNeuronSpiked():
                    neuron.setThreshold(neuron.getThreshold()*0.9)
                    spiked = False
        return spiked

    def BackProp2(self, layer, post, learnRate, deltaPost=None):
        lChanges = []
        deltaList = []
        for i, n in enumerate(layer):
            if deltaPost is None:
                delta = self.deltaOutput(n, post[i])
            else:
                delta = self.deltaHidden(n, post, deltaPost)
            deltaList.append(delta)
            nChanges = []
            for syn in n.getSynapses():
                sChanges = []
                preN = syn.getPre()
                for term in syn.getTerminals():
                        sChanges.append(n.termPot(preN.getLastSpike(), n.getLastSpike(), term.getDelay(), preN.getSign()) * delta * -learnRate)
                nChanges.append(sChanges)
            lChanges.append(nChanges)
        return lChanges, deltaList

    def BackProp(self, layer, post, learnRate, deltaPost=None):
        lChanges = []
        deltaList = []
        for i, n in enumerate(layer):
            if deltaPost is None:
                delta = self.deltaOutput(n, post)
            else:
                delta = self.deltaHidden(n, post, deltaPost)
            deltaList.append(delta)
            nChanges = []
            for syn in n.getSynapses():
                sChanges = []
                preN = syn.getPre()
                preSpikeList = preN.getHistory()
                for term in syn.getTerminals():
                    changes = 0
                    for s, spike in enumerate(n.getHistory()):
                        changes += n.termPot(preSpikeList[s], spike, term.getDelay(), preN.getSign()) * delta * -learnRate
                    sChanges.append(changes)
                nChanges.append(sChanges)
            lChanges.append(nChanges)
        return lChanges, deltaList

    def deltaOutput2(self, n, target):
        a = -(n.getLastSpike() - target)
        b = 0
        for syn in n.getSynapses():
            for term in syn.getTerminals():
                preN = syn.getPre()
                b += term.getWeight()*n.termPotDer(preN.getLastSpike(), n.getLastSpike(), term.getDelay(), preN.getSign())
        if b == 0:
            print("!! Failure In Ouput !!")
            for syn in n.getSynapses():
                preN = syn.getPre()
                for term in syn.getTerminals():
                    print(term.getWeight())
                    print(preN.getLastSpike())
                    print(n.getLastSpike())
                    print(term.getDelay())
                    print("Sign", preN.getSign())
        return a/b

    def deltaOutput(self, n, targets):
        a = 0
        b = 0
        for i, spike in enumerate(n.getHistory()):
            a+= -(spike - targets[i][0])
            for syn in n.getSynapses():
                for term in syn.getTerminals():
                    preN = syn.getPre()
                    b += term.getWeight()*n.termPotDer(preN.getPrevSpike(i), spike, term.getDelay(), preN.getSign())
        return a/b

    def deltaHidden2(self, n, post, deltaPost):
        a = 0
        for i, p in enumerate(post):
            for term in p.getSynapses()[n.getPosition()].getTerminals():
                a += deltaPost[i] * term.getWeight() * p.termPotDer(n.getLastSpike(), p.getLastSpike(), term.getDelay(), n.getSign())
        b = 0
        for syn in n.getSynapses():
            preN = syn.getPre()
            #print(preN.getName())
            for term in syn.getTerminals():
                b += term.getWeight()*n.termPotDer(preN.getLastSpike(), n.getLastSpike(), term.getDelay(), preN.getSign())
        if b == 0:
            self.Failed=True
            #print("!! Failure In Hidden !!")
            #for syn in n.getSynapses():
                #preN = syn.getPre()
                #for term in syn.getTerminals():
                    #print("Terminal Weight:", term.getWeight())
                    #print("Terminal Delay:", term.getDelay())
                    #print("Presynaptic SpikeTime:", preN.getLastSpike())
                    #print("Postsynaptic SpikeTime:", n.getLastSpike())
                    #print("Sign", preN.getSign())
        return a/b


    def deltaHidden(self, n, post, deltaPost):
        a = 0
        b = 0
        for s, spike in enumerate(n.getHistory()):
            for i, p in enumerate(post):
                for term in p.getSynapses()[n.getPosition()].getTerminals():
                    a += deltaPost[i] * term.getWeight() * p.termPotDer(spike, p.getPrevSpike(i), term.getDelay(), n.getSign())
            for syn in n.getSynapses():
                preN = syn.getPre()
                for term in syn.getTerminals():
                    b += term.getWeight()*n.termPotDer(preN.getPrevSpike(s), spike, term.getDelay(), preN.getSign())
        return a/b

    def updateLayerWeights(self, layer, changes):
        for i, n in enumerate(layer):
            for j, s in enumerate(n.getSynapses()):
                for k, t in enumerate(s.getTerminals()):
                    t.updateWeight(changes[i][j][k])


    def MSE(self, target, predicted):
        error = 0
        for i, t in enumerate(target):
            error += (predicted[i] - t) ** 2
        return 0.5*error

    def resetAllSpikes(self):
        for layer in self.layers:
            for neuron in layer:
                neuron.resetSpikes()

    def resetAllThresholds(self):
        for layer in self.layers:
            for neuron in layer:
                neuron.resetThreshold()

    def resetAllHistory(self):
        for layer in self.layers:
            [n.resetSpikeHist() for n in layer]

    def averageNWeights(self):
        print("-- Current Average Weights -- ")
        for layer in self.layers:
            for neuron in layer:
                neuron.getAverageWeight()
                print(neuron.getName(), ":", neuron.getAverageWeight())
    def getAllSpikeTime(self):
        spikeTimes = []
        for layer in self.layers:
            for neuron in layer:
                spikeTimes.append(neuron.getLastSpike())
        return spikeTimes

    def oc(self):
        for x in self.output:
            print("connections:")
            for c in x.getSynapses():
                print(c.getPre().decay)
            isv, term = x.updateISV(x.getLastSpike(), 16)
            print(term)

    def outputWeights(self):
        print("Output Synapse Weights: ")
        for s in self.output[0].getSynapses():
            weight = 0
            maxDelay = np.max([x.getDelay() for x in s.getTerminals()])
            minDelay = np.min([x.getDelay() for x in s.getTerminals()])
            avgDelay = (maxDelay - minDelay)/2
            for t in s.getTerminals():
                if t.getDelay() > avgDelay:
                    weight += t.getWeight()
                elif t.getDelay() < avgDelay:
                    weight -= t.getWeight()
            print(s.getPre().getName(), ":", weight)

    def hiddenWeights(self):
        print("Output Synapse Weights: ")
        for n in self.hidden[0]:
            print("Weights for", n.getName())
            for s in n.getSynapses():
                weight = 0
                maxDelay = np.max([x.getDelay() for x in s.getTerminals()])
                minDelay = np.min([x.getDelay() for x in s.getTerminals()])
                avgDelay = (maxDelay - minDelay)/2
                for t in s.getTerminals():
                    if t.getDelay() > avgDelay:
                        weight += t.getWeight()
                    elif t.getDelay() < avgDelay:
                        weight -= t.getWeight()
                print(s.getPre().getName(), ":", weight)

    def details(self):
        print("Number of hidden layers: ", len(self.hidden))
        print("Input Neurons: ", len(self.input))
        for i, layer in enumerate(self.hidden):
            print("Hidden",i,"Neurons: ", len(layer))
        print("Output Neurons: ", len(self.output))
        print("Total Neurons:", self.neuronNo)
