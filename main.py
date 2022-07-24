import numpy
import struct

import matplotlib.pyplot as plt

from model import Model
from layer import InputLayer, HiddenLayer, OutputLayer

import time

def formatImage(image):
    data = []
    for row in image:
        for pixel in row:
            data.append(pixel)
    return data

def loss(ideal: list, actual: list):
    if len(ideal) == len(actual):
        loss = 0
        for i in range(len(ideal)):
            loss += 0.5 * ((ideal[i] - actual[i]) ** 2)
        return loss
    else: raise Exception("Ideal outputs list is not the same length as actual output list")

# get data from training images file
with open('samples/train-images-idx3-ubyte','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    nrows, ncols = struct.unpack(">II", f.read(8))
    training_images = numpy.fromfile(f, dtype=numpy.dtype(numpy.uint8).newbyteorder('>'))
    training_images = training_images.reshape((size, nrows, ncols))
# get data from testing images labels file
with open('samples/train-labels-idx1-ubyte','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    training_images_labels = numpy.fromfile(f, dtype=numpy.dtype(numpy.uint8).newbyteorder('>'))
    training_images_labels = training_images_labels.reshape((size,))
# get data from testing images file
with open('samples/t10k-images-idx3-ubyte','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    nrows, ncols = struct.unpack(">II", f.read(8))
    testing_images = numpy.fromfile(f, dtype=numpy.dtype(numpy.uint8).newbyteorder('>'))
    testing_images = testing_images.reshape((size, nrows, ncols))
# get data from testing images labels file
with open('samples/t10k-labels-idx1-ubyte','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    testing_images_labels = numpy.fromfile(f, dtype=numpy.dtype(numpy.uint8).newbyteorder('>'))
    testing_images_labels = testing_images_labels.reshape((size,))

model = Model()

model.addLayer(InputLayer(784))
model.addLayer(HiddenLayer(10, "sigmoid"))
model.addLayer(HiddenLayer(20, "sigmoid"))
model.addLayer(OutputLayer(10, "softmax"))

model.initializeWeights()
model.randomizeWeights()

startTimer = time.time()

for i in range(len(testing_images)):

    correctNumber = testing_images_labels[i]
    idealValues = [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
    idealValues[correctNumber - 1] = 1.0

    image = testing_images[i]
    model.inputValues(formatImage(image))
    model.evaluate()

    actualValues = model.outputValues()

    lossValue = loss(idealValues, actualValues)

print("Training 1 model took " + str(time.time() - startTimer) + " seconds")
#plt.imshow(training_images[0,:,:], cmap='gray')
#plt.show()

#print(training_images[0,14,13])