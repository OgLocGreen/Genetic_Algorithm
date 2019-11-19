# coding=utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import gc
import numpy as np 
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from tensorflow.keras import backend as K
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import time
# GPU auf maximal 30% Leistung
"""
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = False
session = tf.Session(config=config)
keras.backend.set_session(session)
"""

import os

def train_and_evalu(gene,dataset = "mnist_fashion",knn_size = "small" ,small_dataset = False ,gpu = False,f1=False):

    if gpu:
        pass
    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    ### Hyperparameter 
    var_learningrate,var_dropout,var_epoch,var_batch_size,optimizer = gene[0],gene[1],gene[2],gene[3],gene[4]

    print("var_learningrate", var_learningrate, "var_dropout", var_dropout, "var_epoch", var_epoch, "var_batch_size", var_batch_size, "optimizer", optimizer)
    fully = False
    cnn = False

    ### Selection of the Dataset
    if dataset == "mnist_fashion": 
        (train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
        fully = True
        label_classes = ["T-shirt/top","Trouser","Pullover","Dress", "Coat", "Sandal", "Shirt","Sneaker","Bag","Ankle_boot"]
    elif dataset == "mnist_digits":
        (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
        fully = True
        label_classes = ["0","1","2", "3", "4", "5","6","7","8 9"]
    elif dataset == "cifar10":
        (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
        train_labels = keras.utils.to_categorical(train_labels, 10)
        test_labels = keras.utils.to_categorical(test_labels, 10)
        train_images = train_images.astype('float32')
        test_images = test_images.astype('float32')
        cnn = True
        label_classes = ["airplane","automobile","bird","cat", "deer", "dog", "frog","horse","ship","truck"]

    train_images = train_images / 255.0
    test_images = test_images / 255.0


    # extra feature um Datenset künstlich zu verkleinern
    small_train_images, small_test_images, small_train_labels, small_test_labels = train_test_split(
        train_images, train_labels, test_size=0.9, shuffle=False)

    ### Selection of the Model
    try:
        tf.compat.v1.set_random_seed(1)
    except:
        tf.random.set_seed(1)
    if fully == True:
        if knn_size == "small":
            model = keras.models.Sequential([
            keras.layers.Flatten(input_shape=train_images.shape[1:]),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(var_dropout),
            keras.layers.Dense(10, activation='softmax')
            ])
        elif knn_size == "medium":
            model = keras.models.Sequential([
            keras.layers.Flatten(input_shape=train_images.shape[1:]),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(var_dropout),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(var_dropout),
            keras.layers.Dense(10, activation='softmax')
            ])
        elif knn_size == "big":
            model = keras.models.Sequential([
            keras.layers.Flatten(input_shape=train_images.shape[1:]),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(var_dropout),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(var_dropout),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(var_dropout),
            keras.layers.Dense(10, activation='softmax')
            ])
    elif cnn == True:
        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(32, (3, 3), padding='same',
                        input_shape=train_images.shape[1:]))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Conv2D(32, (3, 3)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Dropout(var_dropout))

        model.add(keras.layers.Conv2D(64, (3, 3), padding='same'))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Conv2D(64, (3, 3)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Dropout(var_dropout))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(512))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Dropout(var_dropout))
        model.add(keras.layers.Dense(10))
        model.add(keras.layers.Activation('softmax')) 

    ### Optimizer
    adam = keras.optimizers.Adam(lr=var_learningrate)
    Adagrad = keras.optimizers.Adagrad(lr=var_learningrate)
    RMSprop = keras.optimizers.RMSprop(lr=var_learningrate)
    SGD = keras.optimizers.SGD(lr=var_learningrate)
    adadelta = keras.optimizers.Adadelta(learning_rate=var_learningrate)
    adammax = keras.optimizers.Adamax(learning_rate=var_learningrate)
    nadam = keras.optimizers.Nadam(learning_rate=var_learningrate)
    ftrl = keras.optimizers.Ftrl(learning_rate=var_learningrate)
    optimizerarray = [adam, SGD, RMSprop, Adagrad, adadelta,adammax,nadam,ftrl]

    if round(optimizer) < -0.5:
        optimizer = 0
    elif round(optimizer) > 7.5:
        optimizer = 7
        
    if fully == True:
        model.compile(optimizer=optimizerarray[round(optimizer)],
                    loss='sparse_categorical_crossentropy',
                    metrics=['acc'])
    elif cnn == True:
        model.compile(loss='categorical_crossentropy',
              optimizer=optimizerarray[round(optimizer)],
              metrics=['acc'])

    ### Model fit and Evaluate
    mittel = 3
    loss = 0
    acc = 0
    test_loss = 0
    test_acc = 0 
    if small_dataset is True:
        model.fit(small_train_images, small_train_labels, epochs=int(var_epoch),batch_size=int(var_batch_size),use_multiprocessing=True, workers=2,verbose=0)
        for x in range(mittel):
            test_loss, test_acc = model.evaluate(small_test_images, small_test_labels,verbose=0)
            loss += test_loss
            acc += test_acc
    else:
        model.fit(train_images, train_labels, epochs=int(var_epoch),batch_size=int(var_batch_size),use_multiprocessing=True, workers=2,verbose=0)
        for x in range(mittel):
            test_loss,test_acc = model.evaluate(test_images, test_labels,verbose=0)
            loss += test_loss
            acc += test_acc
    test_acc = acc/mittel
    test_loss = loss/mittel

    variables = 0
    variables = np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()])


    if f1:
        if small_dataset:
            y_pred = model.predict(small_test_images)
            y_pred_bool = np.argmax(y_pred, axis=1)
            #print("report",classification_report(test_labels, y_pred_bool))
            precision_score_var = precision_score(small_test_labels, y_pred_bool , average="macro")
            recall_score_var = recall_score(small_test_labels, y_pred_bool , average="macro")
            f1_score_var = f1_score(small_test_labels, y_pred_bool , average="macro")
            print("test_loss: ",test_loss , "test_acc: ", test_acc, "variables",variables, 
            "precision_score_var", precision_score_var,"recall_score_var", recall_score_var,"f1_score_var", f1_score_var)
            cm  = confusion_matrix(y_true = small_test_labels, y_pred = y_pred_bool)
            return test_loss, test_acc, variables, precision_score_var, recall_score_var, f1_score_var
        else:
            y_pred = model.predict(test_images)
            y_pred_bool = np.argmax(y_pred, axis=1)
            print(test_labels,y_pred, y_pred_bool)
            precision_score_var = precision_score(test_labels, y_pred_bool , average="macro")
            recall_score_var = recall_score(test_labels, y_pred_bool , average="macro")
            f1_score_var = f1_score(test_labels, y_pred_bool , average="macro")
            print("test_loss: ",test_loss , "test_acc: ", test_acc, "variables",variables, "precision_score_var", precision_score_var,"recall_score_var", recall_score_var,"f1_score_var", f1_score_var)
            cm  = confusion_matrix(y_true = test_labels, y_pred = y_pred_bool)
            return test_loss, test_acc, variables, precision_score_var, recall_score_var, f1_score_var, cm

    print("test_loss: ",test_loss , "test_acc: ", test_acc, "variables",variables)
    gc.collect()
    return test_loss, test_acc, variables


    
    

if __name__ == "__main__":
    #%%
    #gene = [var_learningrate = 0.05,var_dropout=0.25,var_epoch=100,var_batch_size=16,optimizer=3]
    
    gene= [0.05,0.25,10,16,3]
    start = time.time()
    test_loss, test_acc, variables, precision_score_var, recall_score_var, f1_score_var, cm = train_and_evalu(gene,f1 = True)
    end = time.time() - start


    gene= [0.05,0.25,10,16,3]
    start = time.time()
    test_loss, test_acc, variables, precision_score_var, recall_score_var, f1_score_var, cm = train_and_evalu(gene,dataset = "cifar10",f1 = True)
    end = time.time() - start
    print(end)


    gene= [0.05,0.25,80,16,3]
    start = time.time()
    test_loss, test_acc, variables, precision_score_var, recall_score_var, f1_score_var, cm = train_and_evalu(gene,dataset = "cifar10",f1 = True)
    end = time.time() - start

    #%%
    df_cm = pd.DataFrame(cm)
    plt.figure()
    sn.heatmap(df_cm, annot=True, fmt="d")
    plt.show()

    