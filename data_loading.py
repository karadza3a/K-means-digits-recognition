from mnist import MNIST
import pickle
import numpy as np


def convert(loaded_data, filename):
    images, real_labels = loaded_data
    images = np.array(images)
    real_labels = np.array(real_labels)
    with open(filename, 'wb') as f:
        pickle.dump((images, real_labels), f)

data = MNIST('./mnist_data/')

convert(data.load_training(), 'training.pickle')
convert(data.load_testing(), 'testing.pickle')
