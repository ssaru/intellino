import matplotlib.pyplot as plt
import json
import numpy as np
import csv
import cv2
from pathlib import Path
from torchvision import datasets

"from intellino.core.neuron_cell import NeuronCells"
from intellino.utils.performance import (measure_performance, accuracy, recall, precision)
from neuron_cell import NeuronCells
from mnist_dataset import MNIST
from intellino.utils.data.dataloader import DataLoader

LABELS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def train(number_of_neuron_cell: int = 100,
          length_of_input_vector: int = 256,
          distance_measure: str = "manhattan"):
    train_mnist = MNIST(root="./", train=True, download=False)
    train_dataloader = DataLoader(train_mnist, shuffle=True)

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

    "2. for mnist_dataset MNIST & interpolation image"
    """
    for idx, (data, target) in enumerate(train_dataloader):
        if is_finished:
            break

        image, target = np.array(data).astype(np.uint8), int(target)  # type casting

        image = cv2.resize(data, dsize=(8, 8), interpolation=cv2.INTER_LINEAR)  # resize image
        "image = cv2.resize(image, dsize=(16, 16), interpolation=cv2.INTER_CUBIC)  # resize image"
        "image = cv2.resize(image, dsize=(16, 16))  # resize image"

        print(image)

        image = image.reshape(1, -1).squeeze()  # flatten

        each_clss_distrib[str(target)] += 1
        is_finished = neuron_cells.train(image, target)  # train
    """
    """
    image = image.reshape(16, 16)
    plt.figure()
    plt.imshow(image, cmap = plt.cm.Greys)
    plt.show()
    """

    #print(each_clss_distrib)

    print("FINISHED TRAIN")
    return neuron_cells


def test(neuron_cells: NeuronCells, top_k: int):
    global LABELS
    test_mnist = MNIST(root="./", train=False, download=False)
    test_dataloader = DataLoader(test_mnist, shuffle=True)

    confusion_matrix = np.zeros((len(LABELS), len(LABELS)))

    for idx, (image, target) in enumerate(test_dataloader):
        prediction = neuron_cells.inference(image, top_k)
        confusion_matrix[target][prediction] += 1.

        """
        if is_log:
            print(f"label-> {target}, prediction-> {prediction}")
        """
    return confusion_matrix


if __name__ == "__main__":

    with open("test.csv", "w") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["a", "b", "c"])

    exit()

    # 16KB
    # vector : 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384
    # cell : 16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1
    number_of_neuron_cell = 64
    length_of_input_vector = 256
    distance_measures = ["manhattan"]
    top_k = 5

    number_of_neuron_cells = [2**i for i in range(5, 15)]
    length_of_input_vector = [2**i for i in range(14, 4, -1)]
    hyper_parameter_set = tuple(zip(number_of_neuron_cells, length_of_input_vector))
    top_ks = [i for i in range(1, 25)]
    print(hyper_parameter_set)
    for top_k in top_ks:
        for distance_measure in distance_measures:
            for number_of_neuron_cell, length_of_input_vector in hyper_parameter_set:
                neuron_cells = train(number_of_neuron_cell=number_of_neuron_cell,
                                     length_of_input_vector=length_of_input_vector,
                                     distance_measure=distance_measure)
                confusion_matrix = test(neuron_cells, top_k)
                filename = Path(f"{number_of_neuron_cell}-{length_of_input_vector}-{distance_measure}-{top_k}.json")

                TP, TN, FN, FP = measure_performance(confusion_matrix)

                axis, _ = confusion_matrix.shape
                _accuracy = []
                _recall = []
                _precision = []
                for i in range(axis):
                    _accuracy.append(accuracy(TP[i], TN[i], FN[i], FP[i]))
                    _recall.append(recall(TP[i], FN[i]))
                    _precision.append(precision(TP[i], FP[i]))

                mAcc = str(np.sum(_accuracy) / axis)
                mRecall = str(np.sum(_recall) / axis)
                mPrecision = str(np.sum(_precision) / axis)
                print(f"mean Accuracy : {mAcc}, mean Recall : {mRecall}, mean precision : {mPrecision}")
                exit()
                # with filename.open(mode="w") as json_file:
                #     pass



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


