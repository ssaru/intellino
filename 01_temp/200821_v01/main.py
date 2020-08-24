import matplotlib.pyplot as plt
import numpy as np
import cv2

from torchvision import datasets

"from intellino.core.neuron_cell import NeuronCells"
from neuron_cell import NeuronCells
from mnist_dataset import MNIST
from intellino.utils.data.dataloader import DataLoader


def train():
    train_mnist = MNIST(root="./", train=True, download=False)
    train_dataloader = DataLoader(train_mnist, shuffle=True)

    """
    number_of_neuron_cell = 100
    length_of_input_vector = 256
    """
    number_of_neuron_cell = 64
    length_of_input_vector = 256

    distance_measure = "manhattan"
    neuron_cells = NeuronCells(number_of_neuron_cell, length_of_input_vector, distance_measure)

    each_clss_distrib = {"0": 0,
                         "1": 0,
                         "2": 0,
                         "3": 0,
                         "4": 0,
                         "5": 0,
                         "6": 0,
                         "7": 0,
                         "8": 0,
                         "9": 0, }

    is_finished = False

    "0. for torchvision MNIST"
    """
    for idx, (data, target) in enumerate(train_dataloader):
        if is_finished:
            break

        image, target = np.array(data).astype(np.uint8), int(target)  # type casting
        image = cv2.resize(image, dsize=(16, 16), interpolation=cv2.INTER_CUBIC)  # resize image
        image = image.reshape(1, -1).squeeze()  # flatten
        is_finished = neuron_cells.train(image, target)  # train

        each_clss_distrib[str(target)] += 1
    """
    "1. for mnist_dataset MNIST"
    for idx, (image, target) in enumerate(train_dataloader):
        if is_finished:
            break

        each_clss_distrib[str(target)] += 1
        is_finished = neuron_cells.train(image, target)  # train


    print(each_clss_distrib)

    print("FINISHED TRAIN")
    return neuron_cells


def test(neuron_cells):
    test_mnist = MNIST(root="./", train=False, download=False)
    test_dataloader = DataLoader(test_mnist, shuffle=True)

    TP = 0
    for idx, (image, target) in enumerate(test_dataloader):

        prediction = neuron_cells.inference(image)

        if target == prediction:
            TP += 1

        """
        if is_log:
            print(f"label-> {target}, prediction-> {prediction}")
        """
    print(TP)
    print(len(test_mnist))
    print("Accuracy : {}".format(TP / len(test_mnist)))


if __name__ == "__main__":
    neuron_cells = train()
    test(neuron_cells)

    """
    image, target = np.array(data).astype(np.uint8), int(target)  # type casting
    image = cv2.resize(image, dsize=(16, 16), interpolation=cv2.INTER_CUBIC)  # resize image
    image = image.reshape(1, -1).squeeze()  # flatten
    output = neuron_cells.inference(image)  # inference
    plt.figure()
    plt.imshow(data, cmap=plt.cm.Greys)
    plt.title(f"label-> {target}, result -> {output}")
    plt.show()
    """


