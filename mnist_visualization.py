import matplotlib.pyplot as plt
from torchvision import datasets

train_mnist = datasets.MNIST ('./mnist_data/', download = True, train = True)

test_mnist = datasets.MNIST ('./mnist_data/', download = False, train = False)

for idx , (data, label) in enumerate( train_mnist):
    if idx > 0:
        break
    plt.figure()
    plt.imshow(data, cmap = plt.cm.Greys)
    plt.show()
