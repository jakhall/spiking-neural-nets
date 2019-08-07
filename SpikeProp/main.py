import torch as pt
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from neuron import *
from network import *
import torch.nn.functional as F

class __main__():

    def test1():
        decay = 2
        threshold = 4
        terminals = 10
        noPreCons = 4
        sign = 1
        timesteps = 200
        preCons = []

        print("Pre-Neurons: ")
        for x in range(0, noPreCons - 1):
            preCons.append(Neuron(sign, [], terminals, threshold, decay))
            preCons[x].generateSpike(np.random.rand() * 100)

        preCons.append(Neuron(-1, [], terminals, threshold, decay))
        preCons[3].generateSpike(np.random.rand() * 100)

        postN = Neuron(1, preCons, terminals, threshold, decay)

        history = []
        termH = pt.Tensor(timesteps, noPreCons, terminals)
        print("Post-Neurons: ")
        for t in range(0, timesteps):
            isv, term = postN.updateISV(t, terminals)
            history.append(isv)
            term = pt.Tensor(term)
            termH[t] = term
        fig, ax = plt.subplots()
        colors = ['red', 'blue', 'green', 'orange', 'yellow']
        for n in range(0, termH.shape[1]):
            for t in range(0, termH.shape[2]):
                ax.plot(range(0, timesteps), termH[:, n, t].numpy(), color=colors[n % 5])
        ax.plot(range(0, timesteps), history)
        plt.show()

    def test2():
        decay = 8
        threshold = 4
        terminals = 10
        sign = 1
        timesteps = 200
        layers = [2, 4, 1]
        input = []
        hidden = []
        output = []

        for x in range(0, layers[0]):
            input.append(Neuron(sign, [], terminals, threshold, decay, "inputN" + str(x)))
        for x in range(0, layers[1]):
            hidden.append(Neuron(sign, input, terminals, threshold, decay, "hiddenN" + str(x)))
        for x in range(0, layers[2]):
            output.append(Neuron(sign, hidden, terminals, threshold, decay, "outputN" + str(x)))

        input[0].generateSpike(1)
        input[1].generateSpike(6)
        history = pt.Tensor(timesteps, 3)
        for t in range(0, timesteps):
            for n in input:
                isv, term = n.updateISV(t, terminals)
                history[t, 0] = pt.Tensor([isv])
            for n in hidden:
                isv, term = n.updateISV(t, terminals)
                history[t, 1] = pt.Tensor([isv])
            for n in output:
                isv, term = n.updateISV(t, terminals)
                history[t, 2] = pt.Tensor([isv])


        fig, ax = plt.subplots()
        for n in range(0, history.shape[0]):
                ax.plot(range(0, timesteps), history[:, 0].numpy())
                ax.plot(range(0, timesteps), history[:, 1].numpy())
                ax.plot(range(0, timesteps), history[:, 2].numpy())
        plt.show()


    def test3():

        def graphNeuron(isv, term, neuronNo, its, hidden):
            fig, ax = plt.subplots()
            fig.suptitle("Neuron: " + str(neuronNo) + " Epoch: " + str(its))
            for x in range(0, hidden + 1):
                ax.plot(range(0, 200), isv[:, neuronNo - x].numpy())
            #ax.plot(range(0, 200), isv[:, neuronNo - 5].numpy())
                colour=["red", "blue", "orange", "green", "yellow"]

            for x in range(0, hidden):
                for t in range(0, 16):
                    ax.plot(range(0, 200), term[:, neuronNo, x, t].numpy(), color=colour[x])

            plt.show()

        net = Network([3, 5, 1], 5, 4, 16)
        net.details()

        #data = [[[10, 10], 20], [[1, 1], 20], [[1, 10], 100], [[10, 1], 100]]

        x1 = [[[30, 30], 100]]
        x2 = [[[1, 1], 100]]
        x3 = [[[1, 30], 30]]
        x4 = [[[30, 1], 30]]


        #data = [[[30, 30], [70]], [[1, 1], [85]]]
        #data = [[[30, 1], [100]], [[1, 30], [50]]]
        data = [[[1, 1], [90]], [[1, 30], [70]], [[30, 1], [70]], [[30, 30], [90]]]
        #data = [[[30, 30], [50]], [[1, 1], [80]]]
        #data = [[[1, 1], [60]], [[20, 20], [40]]]
        #data = [[[40], [50]]]
        #data = [[[1, 30], 100]]
        #data2 = [[[30, 1], 30]]

        epoch = 70

        lr = 0.05
        net = Network([3, 5, 1], 5, 4, 16)

        for x in range(0, epoch):
            if (x + 1) % 10 == 0 or x == 0:
                print("Epoch:", x + 1)
                isv, term = net.train(data, lr, 1, 1, True)
                #net.averageNWeights()
                #graphNeuron(isv, term, 8, x + 1, 4)
            else:
                net.train(data,lr, 1, 1)

        print(net.outputWeights())
        print(net.hiddenWeights())

        #data2 = [[[1, 20], 100]]
        #for x in range(0, 1):
            #print("1:", net.train(x1, 0.02))
            #print("2:", net.train(x2, 0.02))
            #print("3:", net.train(x3, 0.02))
            #print("4:", net.train(x4, 0.02))
            #prediction, mse = net.ForwardProp(x4)
            #print(prediction[0])
    test3()
