from numpy import array
from neuron import InputNeuron, HiddenNeuron, OutputNeuron

activationFunctions = ["relu", "sigmoid", "softmax"]

class HiddenLayer:
    def __init__(self, neuronCount = 0, activationFunction = ""):
        self.neuronCount = neuronCount
        self.neurons = []
        for _ in range(neuronCount):
            self.neurons.append(HiddenNeuron())

        if activationFunction.lower() in activationFunctions:
            self.activationFunction = activationFunction.lower()
        else: raise Exception("Unknown activation function type: " + str(activationFunction))

        print("Hidden layer created with " + str(self.neuronCount) + " neurons and activation function " + str(self.activationFunction))

class InputLayer:
    def __init__(self, neuronCount = 0):
        self.neuronCount = neuronCount
        self.neurons = []
        for _ in range(neuronCount):
            self.neurons.append(InputNeuron())

        print("Input layer created with " + str(neuronCount) + " neurons")

class OutputLayer:
    def __init__(self, neuronCount = 0, activationFunction = ""):
        self.neuronCount = neuronCount
        self.neurons = []
        for _ in range(neuronCount):
            self.neurons.append(OutputNeuron())

        if activationFunction.lower() in activationFunctions:
            self.activationFunction = activationFunction.lower()
        else: raise Exception("Unknown activation function type: " + str(activationFunction))

        print("Output layer created with " + str(neuronCount) + " neurons and activation function " + str(self.activationFunction))
    
    def output(self):
        output = []
        for neuron in self.neurons:
            output.append(neuron.value)
        return output