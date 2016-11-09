import numpy as np
import pickle
import time


def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2)


def best_guess(image, model):
    images = np.array([x[1] for x in model])
    i = closest_node(image, images)
    return model[i][0]


def recognize(model, images, real_labels):
    err = 0
    for i in range(len(real_labels)):

        images2 = np.array([x[1] for x in model])
        deltas = images2 - images[i]
        dist_2 = np.einsum('ij,ij->i', deltas, deltas)
        idx = np.argmin(dist_2)
        label = model[idx][0]

        if real_labels[i] != label:
            err += 1

    print("%d/%d => %f" % (err, len(real_labels), float(err) / len(real_labels)))


def main():
    with open('kmeans_testing.pickle', 'rb') as f:
        images, real_labels = pickle.load(f)

    with open('kmeans.pickle', 'rb') as f:
        model = pickle.load(f)

    start_time = time.time()
    recognize(model, images, real_labels)
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()
