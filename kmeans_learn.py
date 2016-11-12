import random

import matplotlib.pyplot as plt
import scipy.cluster
import numpy as np
import pickle

import time


def show_all(images):
    all_digits = np.concatenate([np.reshape(image, (28, 28)) for image in images], axis=1)
    plt.imshow(all_digits, cmap=plt.get_cmap('gray'))
    plt.show()


def main():
    with open('float_training.pickle', 'rb') as f:
        images, real_labels = pickle.load(f)

    start_time = time.time()
    k = np.array([random.choice(images) for _ in range(40)])
    centroid, index_labels = scipy.cluster.vq.kmeans2(images, k, minit='matrix', iter=100)
    print("--- %s seconds ---" % (time.time() - start_time))  #

    show_all(centroid)

    input_labels_string = input()
    input_labels = [int(x) for x in input_labels_string.split(" ")]

    arr = []

    for i in range(len(centroid)):
        if input_labels[i] >= 0:  # allows user to skip a bad centroid
            arr.append((input_labels[i], centroid[i]))

    with open('kmeans.pickle', 'wb') as f:
        pickle.dump(arr, f)


if __name__ == '__main__':
    main()
