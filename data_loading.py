from mnist import MNIST
import pickle


data = MNIST('./mnist_data/')
images, real_labels = data.load_testing()
# images, real_labels = data.load_training()

# with open('training.pickle', 'wb') as f:
with open('testing.pickle', 'wb') as f:
    pickle.dump((images, real_labels), f)
