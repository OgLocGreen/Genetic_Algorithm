import numpy as np
import os
from tensorflow import keras
from time import time
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split
import sklearn
import time
import KNN
import json
import datetime
import plotting
import tools

start_time = time.time()
##zwingen auf cpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

nb_classes = 5


save_file = "{}.{}.{}.json".format(datetime.datetime.now().year,
                                                datetime.datetime.now().month,
                                                datetime.datetime.now().day)

if os.path.isfile(save_file):
    for i in range(1, 10):
        save_file = "{}.{}.{}-{}.json".format(datetime.datetime.now().year,
                                                           datetime.datetime.now().month,
                                                           datetime.datetime.now().day,
                                                           i)
        if os.path.isfile(save_file) == False:
                    break




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





model = KerasClassifier(build_fn=tools.create_model,verbose=1)
n_iter_search = 10 # Number of parameter settings that are sampled.

optimizers = np.array([0, 1, 2])
epochs = np.array([20, 40, 60])
batchsize = np.array([24, 32, 48])
learningrate = np.array([0.0005, 0.001, 0.005])
dropout = np.array([0.05, 0.1, 0.2])

param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batchsize, learningrate=learningrate, dropout=dropout)

random_search = sklearn.model_selection.RandomizedSearchCV(estimator=model,
                                   param_distributions=param_grid,
                                   n_iter=n_iter_search, n_jobs=4)
random_search.fit(small_train_images, small_train_labels)

print("Best: %f using %s" % (random_search.best_score_, random_search.best_params_))
means = random_search.cv_results_['mean_test_score']
stds = random_search.cv_results_['std_test_score']
params = random_search.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

end_time = time.time()
print("finisched in ", end_time-start_time)

best = random_search.best_params_


optimizer = best["optimizer"]
epochs = best["epochs"]
batchsize = best["batch_size"]
learningrate = best["learningrate"]
dropout = best["dropout"]

test_loss, test_acc =  KNN.train_and_evalu(learningrate,dropout,epochs,batchsize,optimizer)

tools.save_params(save_file, means, params)