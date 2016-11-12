import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sknn.mlp import Classifier, Layer
import time


def train(images, real_labels, used_labels):
    scaler = StandardScaler()
    scaler.fit(images)
    images = scaler.transform(images)

    nn = Classifier(layers=[
        Layer("Rectifier", units=100),
        Layer("Rectifier", units=64),
        Layer('Softmax', units=len(used_labels))],
        learning_rate=0.001,
        n_iter=50)
    nn.fit(images, real_labels)
    return nn, scaler


# this will also map each label to index in 'used_labels' list
def filter_by_label(images, real_labels, used_labels):
    images_f = np.array([images[i] for i in range(len(images)) if real_labels[i] in used_labels])
    labels_f = np.array([used_labels.index(label) for label in real_labels if label in used_labels])
    return images_f, labels_f


def train_model(name, used_labels, images, real_labels):
    images, real_labels = filter_by_label(images, real_labels, used_labels)

    start_time = time.time()
    model, scaler = train(images, real_labels, used_labels)
    print("--- %s seconds ---" % (time.time() - start_time))  #

    with open('nn-%s.pickle' % name, 'wb') as f:
        pickle.dump((model, scaler), f)


def main():
    with open('float_training.pickle', 'rb') as f:
        images, real_labels = pickle.load(f)

    train_model("358", [3, 5, 8], images, real_labels)
    train_model("479", [4, 7, 9], images, real_labels)


if __name__ == '__main__':
    main()
