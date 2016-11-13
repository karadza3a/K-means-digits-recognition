import numpy as np
import pickle
import sys
import time

# this will also map each label to index in 'used_labels' list
from utils import no_stdout


def filter_by_label(images, real_labels, used_labels):
    images = np.array([images[i] for i in range(len(images)) if real_labels[i] in used_labels])
    real_labels = np.array([used_labels.index(label) for label in real_labels if label in used_labels])
    return images, real_labels


def test_recognize_part(model, scaler, images, real_labels, used_labels):
    images = scaler.transform(images)

    err = 0
    matrix = np.zeros((len(used_labels), len(used_labels)), dtype=int)
    for i in range(len(real_labels)):
        with no_stdout():
            score = model.predict(np.array([images[i]]))
        label = score[0][0]
        matrix[real_labels[i]][label] += 1
        if real_labels[i] != label:
            err += 1

    np.savetxt(sys.stdout.buffer, matrix, delimiter=',', fmt="%d")
    print("%d/%d => %f" % (err, len(real_labels), float(err) / len(real_labels)))


def test_model(name, used_labels, images, real_labels):
    with open('pickle_data/nn-%s.pickle' % name, 'rb') as f:
        model, scaler = pickle.load(f)

    images, real_labels = filter_by_label(images, real_labels, used_labels)

    start_time = time.time()
    test_recognize_part(model, scaler, images, real_labels, used_labels)
    print("--- %s seconds ---" % (time.time() - start_time))  #


def predict_part(digits, model, scaled_image):
    with no_stdout():
        score = model[0].predict(np.array([scaled_image]))
    label_index = score[0][0]
    return digits[label_index]


def test_combined(images, real_labels, model_kmeans, model358, model479):
    centroids = np.array([x[1] for x in model_kmeans])
    scaled_images_358 = model358[1].transform(images)
    scaled_images_479 = model479[1].transform(images)

    err = 0
    matrix = np.zeros((10, 10), dtype=int)
    for i in range(len(real_labels)):
        deltas = centroids - images[i]
        dist_2 = np.einsum('ij,ij->i', deltas, deltas)
        idx = np.argmin(dist_2)
        label = model_kmeans[idx][0]

        if label in [3, 5, 8]:
            label = predict_part([3, 5, 8], model358, scaled_images_358[i])
        if label in [4, 7, 9]:
            label = predict_part([4, 7, 9], model479, scaled_images_479[i])

        matrix[real_labels[i]][label] += 1
        if real_labels[i] != label:
            err += 1

    np.savetxt(sys.stdout.buffer, matrix, delimiter=',', fmt="%d")
    print("%d/%d => %f" % (err, len(real_labels), float(err) / len(real_labels)))


def main():
    with open('pickle_data/testing.pickle', 'rb') as f:
        images, real_labels = pickle.load(f)
    # test_model("358", [3, 5, 8], images, real_labels)
    # test_model("479", [4, 7, 9], images, real_labels)

    with open('pickle_data/kmeans.pickle', 'rb') as f:
        model_kmeans = pickle.load(f)
    with open('pickle_data/nn-358.pickle', 'rb') as f:
        model358 = pickle.load(f)
    with open('pickle_data/nn-479.pickle', 'rb') as f:
        model479 = pickle.load(f)

    start_time = time.time()
    test_combined(images, real_labels, model_kmeans, model358, model479)
    print("--- %s seconds ---" % (time.time() - start_time))  #


if __name__ == '__main__':
    main()
