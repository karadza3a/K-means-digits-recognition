from mnist import MNIST
import pickle
import numpy as np

data = MNIST('./mnist_data/')
# images, real_labels = data.load_testing()
images, real_labels = data.load_training()
images = np.array(images)
real_labels = np.array(real_labels)

# with open('testing.pickle', 'wb') as f:
with open('training.pickle', 'wb') as f:
    pickle.dump((images, real_labels), f)
