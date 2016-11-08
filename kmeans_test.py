import numpy as np
import pickle
import time


def best_guess(image, model):
    min_error = np.inf
    min_label = -1
    for label, model_image in model:
        mse = ((model_image - image) ** 2).mean()
        if mse < min_error:
            min_error = mse
            min_label = label
    return min_label


def recognize(model, images, real_labels):
    err = 0
    for i in range(len(real_labels)):
        label = best_guess(images[i], model)
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
