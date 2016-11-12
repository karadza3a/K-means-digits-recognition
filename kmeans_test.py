import numpy as np
import pickle
import sys
import time


def recognize(model, images, real_labels):
    centroids = np.array([x[1] for x in model])

    err = 0
    matrix = np.zeros((10, 10), dtype=int)
    for i in range(len(real_labels)):
        deltas = centroids - images[i]
        dist_2 = np.einsum('ij,ij->i', deltas, deltas)
        idx = np.argmin(dist_2)
        label = model[idx][0]

        matrix[real_labels[i]][label] += 1
        if real_labels[i] != label:
            err += 1

    np.savetxt(sys.stdout.buffer, matrix, delimiter=',', fmt="%d")
    print("%d/%d => %f" % (err, len(real_labels), float(err) / len(real_labels)))


def main():
    with open('pickle_data/float_testing.pickle', 'rb') as f:
        images, real_labels = pickle.load(f)

    with open('pickle_data/kmeans.pickle', 'rb') as f:
        model = pickle.load(f)

    start_time = time.time()
    recognize(model, images, real_labels)
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()
