import pickle
from sknn.mlp import Classifier, Layer
import time


def train(images, real_labels):
    nn = Classifier(
        layers=[
            Layer("Rectifier", units=20),
            Layer("Softmax", units=10)
        ],
        learning_rate=0.02,
        n_iter=10)
    nn.fit(images, real_labels)
    return nn


def main():
    with open('training.pickle', 'rb') as f:
        images, real_labels = pickle.load(f)

    start_time = time.time()
    model = train(images, real_labels)
    print("--- %s seconds ---" % (time.time() - start_time))  #

    with open('nn.pickle', 'wb') as f:
        pickle.dump(model, f)

    # testing

    with open('testing.pickle', 'rb') as f:
        images, real_labels = pickle.load(f)

    score = model.score(images, real_labels)
    print(score)

if __name__ == '__main__':
    main()
