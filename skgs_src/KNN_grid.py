from __future__ import absolute_import, division, print_function, unicode_literals

from tensorflow import keras
from sklearn.model_selection import train_test_split
import gc
import tensorflow as tf
"""
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = False
session = tf.Session(config=config)
keras.backend.set_session(session)
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def train_and_evalu(gene,dataset = "mnist_fashion",knn_size = "small" ,small_dataset = False ,gpu = False):
   var_learningrate,var_dropout,var_epoch,var_batch_size,optimizer = gene[0],gene[1],gene[2],gene[3],gene[4]
    #%%
    ### Daten
    print("var_learningrate", var_learningrate, "var_dropout", var_dropout, "var_epoch", var_epoch, "var_batch_size", var_batch_size)
    if dataset == "mnist_fashion": 
        (train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
        fully = True
    elif dataset == "mnist_digits":
        (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
        fully = True
    elif dataset == "cifar10":
        (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
        train_labels = keras.utils.to_categorical(train_labels, 10)
        test_labels = keras.utils.to_categorical(test_labels, 10)
        train_images = train_images.astype('float32')
        test_images = test_images.astype('float32')
        cnn = True

    train_images = train_images / 255.0
    test_images = test_images / 255.0


    #extra feature um Datenset künstlich zu verkleinern
    small_train_images, small_test_images, small_train_labels, small_test_labels = train_test_split(
        train_images, train_labels, test_size=0.9, shuffle=False)

    #%%
    ### Model
    tf.set_random_seed(1)
    model = keras.models.Sequential([
      keras.layers.Flatten(input_shape=(28, 28)),
      keras.layers.Dense(128, activation='relu'),
      keras.layers.Dropout(var_dropout),
      keras.layers.Dense(10, activation='softmax')
    ])


    #%%si
    ### Optimizer




    adam = keras.optimizers.Adam(lr=var_learningrate)
    Adagrad = keras.optimizers.Adagrad(lr=var_learningrate)
    RMSprop = keras.optimizers.RMSprop(lr=var_learningrate)
    SGD = keras.optimizers.SGD(lr=var_learningrate)


    optimizerarray = [adam, RMSprop, SGD, Adagrad]


    if round(optimizer) < 0:
        optimizer = 0
    elif round(optimizer) > 3:
        optimizer = 3

    model.compile(optimizer=optimizerarray[round(optimizer)],
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    #%%
    ### Tensorboard

    #tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

    #%%
    ### Model fit
    #model.fit(small_train_images, small_train_labels, epochs=int(var_epoch),batch_size=int(var_batch_size),use_multiprocessing=True, workers=2)
    model.fit(train_images, train_labels, epochs=int(var_epoch),batch_size=int(var_batch_size),use_multiprocessing=True, workers=2)
    #%%
    ### Model evalu

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print("test_loss: ",test_loss , "test_acc: ", test_acc)
    gc.collect()
    return test_loss, test_acc