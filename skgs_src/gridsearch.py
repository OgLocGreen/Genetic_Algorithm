

import numpy as np
import os
from tensorflow import keras
from time import time





from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import train_test_split

import sklearn


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

nb_classes = 10
(train_images, train_labels), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

small_train_images, small_test_images, small_train_labels, small_test_labels = train_test_split(
    train_images, train_labels, test_size=0.9, shuffle=False)


#X_train = small_train_images.reshape(small_train_images.shape[0], 784)
#X_test = small_test_images.reshape(small_test_images.shape[0], 784)
#X_train = X_train.astype('float32')
#X_test = X_test.astype('float32')
#X_train /= 255
#X_test /= 255


#Y_train = keras.utils.to_categorical(small_train_labels, nb_classes)
#Y_test = keras.utils.to_categorical(small_test_labels, nb_classes)


def create_model(optimizer,learningrate, dropout):
    model = keras.Sequential([
      keras.layers.Flatten(input_shape=(28, 28)),
      keras.layers.Dense(128, activation='relu'),
      keras.layers.Dropout(dropout),
      keras.layers.Dense(10, activation='softmax')
    ])
    adam = keras.optimizers.Adam(lr=learningrate)
    Adagrad = keras.optimizers.Adagrad(lr=learningrate)
    RMSprop = keras.optimizers.RMSprop(lr=learningrate)
    SGD = keras.optimizers.SGD(lr=learningrate)

    optimizerarray = [adam, Adagrad, RMSprop, SGD]

    if round(optimizer) < 0:
        optimizer = 0
    elif round(optimizer) > 3:
        optimizer = 3

    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizerarray[optimizer], metrics=['accuracy'])
    return model

start= time()
model = KerasClassifier(build_fn=create_model,verbose=1)
n_iter_search = 16 # Number of parameter settings that are sampled.

"""
optimizers = np.array([0, 1, 2, 3])
epochs = np.array([20, 40, 60, 80])
batches = np.array([24, 32, 48, 64])
learningrate = np.array([0.0005, 0.001, 0.005, 0.01])
dropout = np.array([0.05, 0.1, 0.2, 0.5])
"""
optimizers = np.array([0, 1, 2])
epochs = np.array([20, 40, 60])
batches = np.array([24, 32, 48])
learningrate = np.array([0.0005, 0.001, 0.005])
dropout = np.array([0.05, 0.1, 0.2])

param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, learningrate=learningrate, dropout=dropout)

random_search = sklearn.model_selection.RandomizedSearchCV(estimator=model,
                                   param_distributions=param_grid,
                                   n_iter=n_iter_search, n_jobs=4)
random_search.fit(small_train_images, small_train_labels)
print("Best: %f using %s" % (random_search.best_score_, random_search.best_params_))
means = random_search.cv_results_['mean_test_score']
stds = random_search.cv_results_['std_test_score']
params = random_search.cv_results_['params']
print("total time:",time()-start)
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
print("finisched")

print(param)