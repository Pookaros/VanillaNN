from pprint import pprint
from perceptron import neuron, synapse, layer
import numpy as np
import MLfunctions as mlf


class Multylayerperceptron:
    # inputs = int # dimention of input vector
    # layers = list[int] # list of #neuron in each layer
    # neurons = np.typing.NDArray # list of layers of with neurons

    def __init__(self, input:int, hidden:list[layer], output:layer, lr:float) -> None:
        self.input = input
        self.hidden = hidden
        self.output = output
        self.neurons = self.init_neurons(lr)
    
    def build_network(self, input:layer, hidden:list[layer], output:layer, lr:float) -> None:
        self.input = input
        self.hidden = hidden
        self.output = output
        self.neurons = self.init_neurons(lr)
    
    def init_neurons(self, lr:float):
        #rows = max(self.layers)
        columns = len(self.layers)
        self.neurons = [[] for _ in range(columns)]
        for j in range(columns):
            for i in range(self.layers[j]):
                if j == 0:
                    self.neurons[j].append(neuron(0, self.inputs, lr))
                else:
                    self.neurons[j].append(neuron(0, self.layers[j-1], lr))
        return self.neurons
    
    #forward propagation function (I only made it to collect various data from the MLP)
    def forward_prop(self, inp:np.typing.ArrayLike):
        columns = len(self.layers)
        #rows = max(self.layers)
        Z, A = [[] for _ in range(columns)], [[] for _ in range(columns)]
        for j in range(columns):
            for i in range(self.layers[j]):
                neuron = self.neurons[j][i]
                if j == 0: # for every neuron before the last one use sigmoid
                    Z[j].append(neuron.estimator(inp))
                    A[j].append(neuron.guess(inp, mlf.sig))
                else: #on the last one we use soft max
                    Z[j].append(neuron.estimator(A[j-1]))
            if j > 0: A[j] = mlf.softmax(Z[j])
        return Z, A
    
    def backward_prop(self, estimation: list[float], guess: list[float], tar, inp):
        columns = len(self.layers)
        #rows = max(self.layers)
        for j in reversed(range(columns)): # going from last layer to starting layer
            for i in range(self.layers[j]):
                neuron = self.neurons[j][i]
                if j == columns - 1: #case 0 (from outputs)
                    #tempweight = [a.weights[i] for a in self.neurons[j-1]]
                    #print("PRE:  ", guess[j-1] + [1], tar[i])
                    neuron.temperror = guess[j][i] - tar[i]
                    #print("AFTER:  ", neuron.weights, neuron.lr, neuron.temperror)
                    neuron.trainsize += 1
                    neuron.Update_weights(guess[j-1] + [1])
                elif j > 0:
                    #print("GUESS:   ", guess[j-1] + [1], tar[i])
                    weightlist = [x.weights[i] for x in self.neurons[j+1]]
                    inner = mlf.innerprod(weightlist, guess[j+1])
                    #print(self.neurons[j+1][i].temperror, weightlist, inner)
                    neuron.temperror = inner * mlf.sig_deriv(estimation[j][i])
                    #print("AFTER:  ", neuron.weights, neuron.lr, neuron.temperror, guess[j])
                    neuron.trainsize += 1
                    neuron.Update_weights(guess[j-1] + [1])
                else:
                    #print("GUESS:   ", inp, tar[i])
                    weightlist = [x.weights[i] for x in self.neurons[j+1]]
                    inner = mlf.innerprod(weightlist, guess[j+1])
                    #print(self.neurons[j+1][i].temperror, weightlist, inner)
                    #print(self.neurons[j+1][i].weights * self.neurons[j+1][i].temperror * mlf.ReLU_deriv(estimation[j][i]))
                    neuron.temperror = inner * mlf.sig_deriv(estimation[j][i])
                    #print("AFTER:  ", neuron.weights, neuron.lr, neuron.temperror, guess[j])
                    neuron.trainsize += 1
                    neuron.Update_weights(inp + [1])

    def get_predictions(inp):
        return np.argmax(inp, 0)
    
    def gradient_descent(self, inp, tar, iterations):
        acc = 0
        size = 1
        for i in range(iterations):
            single_input = inp[i].tolist()
            Z, A = self.forward_prop(single_input)
            one_hot_Y = mlf.one_hot(tar)
            pprint(Z)
            pprint(A)
            self.backward_prop(Z, A, one_hot_Y[:,i], single_input)
            if i % 10 == 0:
                print("Iteration: ", i)
                predictions = np.argmax(A[-1],0)
                print("Predicted Group:  ", predictions, "Actual Group: ", tar[i])
                if predictions == tar[i]:
                    acc += 1
                size += 1
        print(acc,size)