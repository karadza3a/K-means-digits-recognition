import pickle
from sknn.mlp import Classifier, Layer
from sklearn.preprocessing import StandardScaler
import time


def train(images, real_labels):
    scaler = StandardScaler()
    scaler.fit(images)
    images = scaler.transform(images)

    nn = Classifier(layers=[
        Layer("Rectifier", units=100),
        Layer("Rectifier", units=64),
        Layer('Softmax', units=10)],
        learning_rate=0.001,
        n_iter=50)
    nn.fit(images, real_labels)
    return nn, scaler


def main():
    with open('pickle_data/training.pickle', 'rb') as f:
        images, real_labels = pickle.load(f)

    start_time = time.time()
    model, scaler = train(images, real_labels)
    print("--- %s seconds ---" % (time.time() - start_time))  #

    with open('pickle_data/nn.pickle', 'wb') as f:
        pickle.dump((model, scaler), f)


if __name__ == '__main__':
    main()
