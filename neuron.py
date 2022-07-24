class HiddenNeuron:
    def __init__(self):
        self.value = 0.0
        # weights refers to the number of weights this
        # neuron has in connections to the previous layer
        self.weights = []

class InputNeuron:
    def __init__(self):
        self.value = 0.0

class OutputNeuron:
    def __init__(self):
        self.value = 0.0
        # weights refers to the number of weights this
        # neuron has in connections to the previous layer
        self.weights = []