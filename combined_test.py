import numpy as np
import pickle
import sys
import time


# this will also map each label to index in 'used_labels' list
def filter_by_label(images, real_labels, used_labels):
    images = np.array([images[i] for i in range(len(images)) if real_labels[i] in used_labels])
    real_labels = np.array([used_labels.index(label) for label in real_labels if label in used_labels])
    return images, real_labels


def recognize_part(model, scaler, images, real_labels, used_labels):
    images = scaler.transform(images)

    err = 0
    matrix = np.zeros((len(used_labels), len(used_labels)), dtype=int)
    for i in range(len(real_labels)):
        score = model.predict(np.array([images[i]]))
        label = score[0][0]
        matrix[real_labels[i]][label] += 1
        if real_labels[i] != label:
            err += 1

    np.savetxt(sys.stdout.buffer, matrix, delimiter=',', fmt="%d")
    print("%d/%d => %f" % (err, len(real_labels), float(err) / len(real_labels)))


def test_model(name, used_labels, images, real_labels):

    with open('nn-%s.pickle' % name, 'rb') as f:
        model, scaler = pickle.load(f)

    images, real_labels = filter_by_label(images, real_labels, used_labels)

    start_time = time.time()
    recognize_part(model, scaler, images, real_labels, used_labels)
    print("--- %s seconds ---" % (time.time() - start_time))  #


def main():
    with open('float_testing.pickle', 'rb') as f:
        images, real_labels = pickle.load(f)
    test_model("358", [3, 5, 8], images, real_labels)
    test_model("249", [2, 4, 9], images, real_labels)


if __name__ == '__main__':
    main()
