import pickle
import time


def main():
    with open('nn.pickle', 'rb') as f:
        model = pickle.load(f)

    with open('testing.pickle', 'rb') as f:
        images, real_labels = pickle.load(f)

    start_time = time.time()
    score = model.score(images, real_labels)
    print("--- %s seconds ---" % (time.time() - start_time))  #
    print(score)

if __name__ == '__main__':
    main()
