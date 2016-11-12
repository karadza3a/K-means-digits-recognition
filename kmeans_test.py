import numpy as np
import pickle
import time


# def closest_node(node, nodes):
#     nodes = np.asarray(nodes)
#     deltas = nodes - node
#     dist_2 = np.einsum('ij,ij->i', deltas, deltas)
#     return np.argmin(dist_2)
#
#
# def best_guess(image, model):
#     images = np.array([x[1] for x in model])
#     i = closest_node(image, images)
#     return model[i][0]
#
def incorrect(label1, label2):
    if label1 == label2:
        return False
    # brackets = [
    #     [3, 5, 8],
    #     [4, 7, 9],
    # ]
    # for bracket in brackets:
    #     if label1 in bracket and label2 in bracket:
    #         return False
    return True


def recognize(model, images, real_labels):
    err = 0
    images2 = np.array([x[1] for x in model])

    # matrix = np.zeros((10, 10), dtype=int)
    for i in range(len(real_labels)):
        deltas = images2 - images[i]
        dist_2 = np.einsum('ij,ij->i', deltas, deltas)
        idx = np.argmin(dist_2)
        label = model[idx][0]

        # matrix[real_labels[i]][label] += 1
        if incorrect(real_labels[i], label):
            err += 1

    print("%d/%d => %f" % (err, len(real_labels), float(err) / len(real_labels)))
    # import sys
    # np.savetxt(sys.stdout.buffer, matrix, delimiter=',', fmt="%d")


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
