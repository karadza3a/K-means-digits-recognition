from mnist import MNIST
import pickle
import numpy as np

import combined_learn
import kmeans_learn
import nn_learn


def convert(loaded_data, filename):
    images, real_labels = loaded_data
    images = [[float(x) for x in image] for image in images]  # needed for kmeans data (cause of Scipy)
    images = np.array(images)
    real_labels = np.array(real_labels)
    with open(filename, 'wb') as f:
        pickle.dump((images, real_labels), f)


def main():
    print("Converting data. (1/4)")
    data = MNIST('./mnist_data/')
    convert(data.load_training(), 'pickle_data/training.pickle')
    convert(data.load_testing(), 'pickle_data/testing.pickle')
    print("Training K-means. (2/4)")
    kmeans_learn.main()
    print("Training NN. This will take a while. (3/4)")
    nn_learn.main()
    print("Training combo. This will take a while. (4/4)")
    combined_learn.main()
    print("Done.")


if __name__ == '__main__':
    main()
