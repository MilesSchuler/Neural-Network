from layer import InputLayer, HiddenLayer, OutputLayer
from typing import Union
import math
import random

class Model:
    def __init__(self):
        self.inputLayer = InputLayer
        self.outputLayer = OutputLayer
        self.hiddenLayers = []

    def addLayer(self, layer: Union[InputLayer, HiddenLayer, OutputLayer]):
        if type(layer) == InputLayer:
            self.inputLayer = layer
        elif type(layer) == HiddenLayer:
            self.hiddenLayers.append(layer)
        elif type(layer) == OutputLayer:
            self.outputLayer = layer

    def inputValues(self, values: list):
        if len(values) < self.inputLayer.neuronCount:
            raise Exception("Not enough input values for each neuron")
        if len(values) > self.inputLayer.neuronCount:
            raise Exception("Too many input values for each neuron")
        for i in range(0, len(values)):
            self.inputLayer.neurons[i].value = values[i]

    def outputValues(self):
        return self.outputLayer.output()

    def initializeWeights(self):
        # search for connections between layers
        for i in range(0, len(self.hiddenLayers)):
            # check if it is connected to input layer
            if i == 0:
                for neuron in self.hiddenLayers[0].neurons:
                    neuron.weights = [0] * self.inputLayer.neuronCount
            else:
                for neuron in self.hiddenLayers[i].neurons:
                    neuron.weights = [0] * self.inputLayer.neuronCount
        # initialize weights for output layer
        for neuron in self.outputLayer.neurons:
            neuron.weights = [0] * self.hiddenLayers[-1].neuronCount

    def randomizeWeights(self):
        # randomize hidden layer weights
        for layer in self.hiddenLayers:
            for neuron in layer.neurons:
                for i in range(0, len(neuron.weights)):
                    neuron.weights[i] = random.uniform(0, 1)
        # randomize output layer weights
        for neuron in self.outputLayer.neurons:
            for i in range(0, len(neuron.weights)):
                neuron.weights[i] = random.uniform(0, 1)

    def evaluate(self):
        for i in range(0, len(self.hiddenLayers)):
            # check to see if it is connected to input layer
            if i == 0:
                # hidden layer connected to input layer
                for hiddenNeuron in self.hiddenLayers[i].neurons:
                    sum = 0
                    for j in range(0, self.inputLayer.neuronCount):
                        sum += self.inputLayer.neurons[j].value * hiddenNeuron.weights[j]
                    hiddenNeuron.value = globals()[self.hiddenLayers[i].activationFunction](sum) if self.hiddenLayers[i].activationFunction != "softmax" else sum # sigmoid(sum)
            else:
                # hidden layer connected to hidden layer
                for hiddenNeuron in self.hiddenLayers[i].neurons:
                    sum = 0
                    for j in range(0, self.hiddenLayers[i - 1].neuronCount):
                        sum += self.hiddenLayers[i - 1].neurons[j].value * hiddenNeuron.weights[j]
                    hiddenNeuron.value = globals()[self.hiddenLayers[i].activationFunction](sum) if self.hiddenLayers[i].activationFunction != "softmax" else sum # sigmoid(sum)
            
            # special code for handling softmax function for hidden layer
            if self.hiddenLayers[i].activationFunction == "softmax":
                values = []
                for neuron in self.hiddenLayers[i].neurons:
                    values.append(neuron.value)
                softmaxValues = softmax(values)
                for j in range(self.hiddenLayers.neuronCount):
                    self.hiddenLayers[i].neurons[j].value = softmaxValues[j]

        # output layer connected to hidden layer
        for outputNeuron in self.outputLayer.neurons:
            sum = 0.0
            for i in range(0, len(outputNeuron.weights)):
                sum += self.hiddenLayers[-1].neurons[i].value * outputNeuron.weights[i]
            outputNeuron.value = globals()[self.outputLayer.activationFunction](sum) if self.outputLayer.activationFunction != "softmax" else sum # sigmoid(sum)

        # special code for handling softmax function for hidden layer
        if self.outputLayer.activationFunction == "softmax":
            values = []
            for neuron in self.outputLayer.neurons:
                values.append(neuron.value)
            softmaxValues = softmax(values)
            for j in range(self.outputLayer.neuronCount):
                self.outputLayer.neurons[j].value = softmaxValues[j]
    
def relu(value):
    return max(0.0, value)

def sigmoid(value):
    return 1.0 / (1.0 + math.exp(-value))

def softmax(vector: list):
    expList = []
    sum = 0
    for item in vector:
        expList.append(math.e ** item)
        sum += math.e ** item
    dividedExpList = []
    for item in expList:
        dividedExpList.append(item / sum)
    return dividedExpList
# loss is 100% minus whatever percentage of the end neurons were activated that were not the given digit
# check loss of a model by running through testing data and averaging losses