
import matplotlib.pyplot as plt

from intellino.core.neuron_cell import NeuronCells
from intellino.utils.data.dataloader import DataLoader

from mnist_dataset import MNIST


def test(neuron_cells, is_log=False):
    test_mnist = MNIST(root="./", train=False, download=False)
    test_dataloader = DataLoader(test_mnist, shuffle=True)

    TP = 0
    for idx, (image, target) in enumerate(test_dataloader):

        prediction = neuron_cells.inference(image)

        if target == prediction:
            TP += 1

        if is_log:
            print(f"label-> {target}, prediction-> {prediction}")

    print("Accuracy : {}".format(TP / len(test_mnist)))


def train():
    train_mnist = MNIST(root="./", train=True, download=False)
    train_dataloader = DataLoader(train_mnist, shuffle=True)

    number_of_neuron_cells = 100
    length_of_input_vector = 256
    neuron_cells = NeuronCells(number_of_neuron_cells=number_of_neuron_cells,
                               length_of_input_vector=length_of_input_vector,
                               measure="manhattan")

    print(f"Number of Cells : {len(neuron_cells)}")

    each_clss_distrib = {"0": 0,
                         "1": 0,
                         "2": 0,
                         "3": 0,
                         "4": 0,
                         "5": 0,
                         "6": 0,
                         "7": 0,
                         "8": 0,
                         "9": 0,}

    finish_train = False

    print("Start Training!")
    for idx, (image, target) in enumerate(train_dataloader):
        if finish_train:
            break

        each_clss_distrib[str(target)] += 1

        finish_train = neuron_cells.train(vector=image, target=target)

    print("Done!")
    return neuron_cells


if __name__ == "__main__":

    # Dataset Visualization
    mnist_tets = MNIST(root="./", train=True, download=False)
    image, target = mnist_tets.__getitem__(0)
    image = image.reshape(16, 16)
    plt.figure()
    plt.imshow(image)
    plt.title(str(target))
    plt.show()

    # Train NeuronCells
    neuron_cells = train()
    test(neuron_cells, is_log=False)






