import matplotlib.pyplot as plt
import numpy as np
import cv2

from torchvision import datasets
"from intellino.core.neuron_cell import NeuronCells"
from neuron_cell import NeuronCells


def train():
    train_mnist = datasets.MNIST('./mnist_data/', download=True, train=True)

    number_of_neuron_cell = 100
    length_of_input_vector = 256
    distance_measure = "manhattan"
    neuron_cells = NeuronCells(number_of_neuron_cell, length_of_input_vector, distance_measure)

    is_finished = False
    for idx, (data, target) in enumerate(train_mnist):
        if is_finished:
            break

        image, target = np.array(data).astype(np.uint8), int(target)  # type casting
        image = cv2.resize(image, dsize=(16, 16), interpolation=cv2.INTER_CUBIC)  # resize image
        image = image.reshape(1, -1).squeeze()  # flatten
        is_finished = neuron_cells.train(image, target)  # train

    print("FINISHED TRAIN")
    return neuron_cells


def test(neuron_cells):
    test_mnist = datasets.MNIST('./mnist_data/', download=True, train=False)
    for idx, (data, target) in enumerate(test_mnist):
        if idx > 9:
            break
        image, target = np.array(data).astype(np.uint8), int(target)  # type casting
        image = cv2.resize(image, dsize=(16, 16), interpolation=cv2.INTER_CUBIC)  # resize image
        image = image.reshape(1, -1).squeeze()  # flatten
        output = neuron_cells.inference(image)  # inference

        plt.figure()
        plt.imshow(data, cmap=plt.cm.Greys)
        plt.title(f"label-> {target}, result -> {output}")
        plt.show()


if __name__ == "__main__":
    neuron_cells = train()
    test(neuron_cells)

