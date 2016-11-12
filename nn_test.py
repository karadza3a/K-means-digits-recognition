import numpy as np
import pickle
import time
import sys

from utils import no_stdout


def recognize(model, scaler, images, real_labels):
    images = scaler.transform(images)

    err = 0
    matrix = np.zeros((10, 10), dtype=int)
    for i in range(len(real_labels)):
        with no_stdout():
            score = model.predict(np.array([images[i]]))
        label = score[0][0]
        matrix[real_labels[i]][label] += 1
        if real_labels[i] != label:
            err += 1

    np.savetxt(sys.stdout.buffer, matrix, delimiter=',', fmt="%d")
    print("%d/%d => %f" % (err, len(real_labels), float(err) / len(real_labels)))


def main():
    with open('pickle_data/nn.pickle', 'rb') as f:
        model, scaler = pickle.load(f)

    with open('pickle_data/testing.pickle', 'rb') as f:
        images, real_labels = pickle.load(f)

    start_time = time.time()
    recognize(model, scaler, images, real_labels)
    print("--- %s seconds ---" % (time.time() - start_time))  #


if __name__ == '__main__':
    main()
