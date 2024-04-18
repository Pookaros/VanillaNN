import pprint
from scipy.misc import derivative
import random
import pandas as pd
import numpy as np
import numpy.typing as nmp
import MLfunctions as mlf
from typing import List, Callable

from itertools import product
import os


class neuron:
    # weights:List[float] #the weights of the neuron (till further training)
    # lr:int #learning rate of neuron 0<lr<2 but it should not exceed 1
    # bias:float #bias of the hyperplane
    # trainsize:int #number of inputs used to train it (till further training)
    # temperror:float # last estimation error of the neuron
    # tempguess:float # the last guess of th neuron

    def __init__(self, lr:float, bias:float() = 0.0, _activation_method:Callable[[int], str] = mlf.tanh) -> None:
        self.bias = bias
        self.lr = lr
        self.value = 0 
        self.temperror = 0
        self.tempguess = 0
        self.activation_method = _activation_method

    def preview(self):
        name = repr(self)[-4:-1]
        return [name, self.lr, self.tempguess, self.temperror, self.activation_method]
    
    def guess(self):
        self.value = self.interference()
        self.tempguess = self.activation_method(self.value + self.bias)
        return self.tempguess
    
    def reconsider(self):
        self.temperror = derivative(self.activation_method, self.value, dx=1e-6) * self.interference()

    def update_weights(self):
        [_synapse.update_weight() for _synapse in self.get_front_synapses()]

    def get_synapses(self) -> List['synapse']:
        return [synapse for synapse in synapse.get_all() if (synapse.back_neuron or synapse.front_neuron) == self]
    
    def get_front_synapses(self) -> List['synapse']:
        return [_synapse for _synapse in synapse.get_all() if (_synapse.back_neuron) == self]
    
    def get_back_synapses(self) -> List['synapse']:
        return [synapse for synapse in synapse.get_all() if (synapse.front_neuron) == self]
    
    # SUM OF SIGNALS WHEN THEY REACH ON NEURON AND THEN DISSAPEAR
    def interference(self) -> float:
        intereference = sum([_signal.strength for _signal in self.signals_in()])
        [_signal.destroy() for _signal in self.signals_in()]
        return intereference
        
    
    def signals_in(self) -> List['signal']:
        return [_signal for _signal in signal.get_all() if _signal.destination == self]
    
    def signals_out(self) -> List['signal']:
        return [_signal for _signal in signal.__instances if _signal.origin == self]


class synapse:
    __instances:List['synapse'] = [] # List that contains all the synapses

    def __init__(self, back_neuron:neuron, weight:float, front_neuron:neuron) -> None:
        if not synapse.exists(back_neuron, front_neuron):
            self.back_neuron = back_neuron
            self.weight = weight
            self.front_neuron = front_neuron
            synapse.__instances.append(self) # append to the List of all synapses when created
        else: print(f"Synapse already exists")
    
    def destroy(self):
        synapse.__instances.remove(self)
        del self
    
    def exists(back_neuron:neuron, front_neuron:neuron) -> bool:
        synapses = synapse.get_all()
        if (back_neuron,front_neuron) in [(synapse.back_neuron, synapse.front_neuron) for synapse in synapses]: return True
        return False

    def preview(self):
        back_neuron_name = repr(self.back_neuron)[-4:-1]
        _synapse_weight = self.weight
        front_neuron_name = repr(self.front_neuron)[-4:-1]
        return [back_neuron_name, _synapse_weight, front_neuron_name]

    def get_all() -> List['synapse']:
        return synapse.__instances

    def reconnect_back_neuron(self, new_back_neuron:neuron) -> None:
        self.back_neuron = new_back_neuron
    
    def recconect_front_neuron(self, new_front_neuron:neuron) -> None:
        self.front_neuron = new_front_neuron
    
    def get_front_neurons() -> List['neuron']:
        return list(set([_synapse.front_neuron for _synapse in synapse.get_all()]))
    
    def get_back_neurons() -> List['neuron']:
        return list(set([_synapse.back_neuron for _synapse in synapse.get_all()]))
    
    def get_weights(synapses:List['synapse']) -> List[float]:
        return [_synapse.weight for _synapse in synapses]
    
    def propagate_forward(self):
        signal(_synapse = self, reversed=False)
    
    def propagate_backward(self):
        signal(_synapse = self, reversed=True)
    
    def update_weight(self):
        self.weight += self.back_neuron.lr * self.back_neuron.tempguess * self.front_neuron.temperror
        

class signal:
    __instances:List['signal'] = []

    def __init__(self, _synapse:synapse, reversed:bool = False) -> None:
        self.synapse = _synapse
        if reversed:
            self.origin = self.synapse.front_neuron
            self.destination = self.synapse.back_neuron
            self.strength = self.synapse.weight * self.origin.temperror
        else: 
            self.origin = self.synapse.back_neuron
            self.destination = self.synapse.front_neuron
            self.strength = self.synapse.weight * self.origin.tempguess

        signal.__instances.append(self)
    
    def preview(self):
        _origin = repr(self.origin)[-4:-1]
        _strength = self.strength
        _destination = repr(self.destination)[-4:-1]
        return [_origin, _strength, _destination]
    
    def destroy(self) -> None:
        signal.__instances.remove(self)
        del self
    
    def get_all() -> List['signal']:
        return signal.__instances


class group:
    __instances: List['group'] = []

    def __init__(self, neurons: List['neuron']) -> None:
        self.neurons = neurons
        group.__instances.append(self)
    
    def destroy(self) -> None:
        group.__instances.remove(self)
        del self
    
    @classmethod
    def get_all(cls) -> List['synapse']:
        return cls.__instances
    
    @classmethod
    def spawn(cls, size:int, lr:float = 4, bias:float = 0.0):
        neurons = [neuron(lr,bias) for _ in range(size)]
        group = cls(neurons)
        return group
    
    def add_neuron(self, neuron:neuron) -> None:
        self.neurons.append(neuron)
    
    def add_neurons(self, neurons:List[neuron]) -> None:
        self.neurons.extend(neurons)
    
    def merge_groups(group_1:'group', group_2:'group') -> 'group':
        group_1.add_neurons(group_2.neurons)
        group_2.destroy()
        return group_1
    
    def connect_groups(group_1:'group', group_2:'group'):
        return [synapse(neuron_1, random.choice([-1, 1]), neuron_2) for (neuron_1 , neuron_2) in product(group_1.neurons, group_2.neurons)]
    
    def get_synapses(self) -> List[synapse]:
        return [_synapse for _neuron in self.neurons for _synapse in _neuron.get_synapses()]
    
    def get_front_synapses(self) -> List[synapse]:
        return [_synapse for _neuron in (self.neurons and synapse.get_back_neurons()) for _synapse in _neuron.get_synapses()]
    
    def get_back_synapses(self) -> List[synapse]:
        return [_synapse for _neuron in (self.neurons and synapse.get_front_neurons()) for _synapse in _neuron.get_synapses()]
    
    def signals_out_forward(self):
        [_synapse.propagate_forward() for _synapse in self.get_front_synapses()]
    
    def signals_out_backward(self):
        [(_synapse.propagate_backward()) for _synapse in self.get_back_synapses()]
    
    def signals_in_forward(self):
        [(_neuron.guess()) for _neuron in self.neurons]
    
    def signals_in_backward(self):
        [_neuron.update_weights() for _neuron in self.neurons]
    

class layer(group):
    __instances: List['layer'] = []
    __tags = ['input', 'output', 'hidden']

    def __init__(self, neurons:List[neuron]) -> None:
        # Check if the provided tag is in the List of allowed tag
        self.neurons = neurons
        layer.__instances.append(self)
    
    def destroy(self) -> None:
        layer.__instances.remove(self)
        del self
    
    def get_all() -> List['synapse']:
        return layer.__instances
    
    def preview(self):
        return [repr(_neuron)[-4:-1] for _neuron in self.neurons]

    @classmethod
    def spawn(cls, size:int, lr:float = 4.0, bias:float = 0.0) -> 'layer':
        return super().spawn(int(size), lr, bias)

    @classmethod
    def merge_layers(cls, layer_1:'layer', layer_2:'layer') -> 'layer':
        return super().merge_groups(layer_1, layer_2)
    
    @classmethod
    def connect_layers(cls, layer_1:'layer', layer_2:'layer'):
        return super().connect_groups(layer_1, layer_2)
    
    def get_synapses(self) -> List[synapse]:
        return super().get_synapses()
    
    def get_front_synapses(self) -> List[synapse]:
        return super().get_front_synapses()

    def get_back_synapses(self) -> List[synapse]:
        return super().get_back_synapses()
    
    def signals_in_forward(self):
        return super().signals_in_forward()
    
    def signals_out_forward(self):
        return super().signals_out_forward()

    def signals_in_backward(self):
        return super().signals_in_backward()
    
    def signals_out_backward(self):
        return super().signals_out_backward()
    

class mlp:

    #multy layer perceptron
    def __init__(self, inp:layer, out:layer, hidden:List[layer] = []) -> None:
        self.input_layer = inp
        self.hidden_layer = hidden
        self.output_layer = out
        self.layers:List[layer] = []
        self.layers.extend(hidden)
        self.layers.append(out)
        self.layers.insert(0, inp)
        self.trainsize = 0
        self.synapses = [layer.connect_layers(_layer, self.layers[i+1]) for (i,_layer) in enumerate(self.layers[:-1])]
    
    def propagate_forward(self, _inputs:List[float]):
        for _input,_neuron in zip(_inputs, self.input_layer.neurons): # we load the inputs in the input layer
            _neuron.tempguess = _input
        [(self.layers[i].signals_out_forward(), self.layers[i+1].signals_in_forward()) for (i, _layer) in enumerate(self.layers[:-1])]
        return self.output()

    def propagate_backward(self, _expecations):
        for _expectation,_neuron in zip(_expecations, self.output_layer.neurons): # we load the errors in the input layer
            _neuron.temperror = ((_expectation -  _neuron.tempguess)**2) / 2

        # send signals backward -> signals in the back layer -> each neuron weight get updated
        for i in range(len(self.layers) - 1, 0, -1):
            self.layers[i].signals_out_backward()
            self.layers[i - 1].signals_in_backward()
            for _neuron in self.layers[i].neurons:
                _neuron.update_weights()

        self.trainsize += 1 #we count the times the model was trained

    def output(self, method:Callable[[mlf.k_fold, int], List[float]] = mlf.k_fold.softmax) -> List[float]:
        Z = [_neuron.tempguess for _neuron in self.output_layer.neurons]
        for (_neuron, _estimation) in zip(self.output_layer.neurons, method(Z)):
            _neuron.tempguess = _estimation
        return [_neuron.tempguess for _neuron in self.output_layer.neurons]
    
    def guess(self) -> List[float]:
        my_list = [_neuron.tempguess for _neuron in self.output_layer.neurons]
        max_index = my_list.index(max(my_list))
        result =  [0 for _ in my_list]
        result[max_index] = 1
        return result





