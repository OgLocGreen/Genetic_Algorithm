from tensorflow import keras
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import json
import tensorflow as tf
import sys
import socket

sys.path.append('../')
from src_evaluation.evaluation import write_cell

def create_model(optimizer,learningrate, dropout):
    try:
        tf.set_random_seed(1)
    except:
        tf.random.set_seed(1)
    model = keras.Sequential([
      keras.layers.Flatten(input_shape=(28, 28)),
      keras.layers.Dense(128, activation='relu'),
      keras.layers.Dropout(rate = dropout),
      keras.layers.Dense(10, activation='softmax')
    ])
    adam = keras.optimizers.Adam(lr=learningrate)
    Adagrad = keras.optimizers.Adagrad(lr=learningrate)
    RMSprop = keras.optimizers.RMSprop(lr=learningrate)
    SGD = keras.optimizers.SGD(lr=learningrate)
    adadelta = keras.optimizers.Adadelta(learning_rate=learningrate)
    adammax = keras.optimizers.Adamax(learning_rate=learningrate)
    nadam = keras.optimizers.Nadam(learning_rate=learningrate)

    optimizerarray = [adam, SGD, RMSprop, Adagrad, adadelta,adammax,nadam]

    if round(optimizer) < -0.5:
        optimizer = 0
    elif round(optimizer) > 6.5:
        optimizer = 6

    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizerarray[optimizer], metrics=['accuracy'])
    return model

def create_model_small(optimizer,learningrate, dropout):
    try:
        tf.compat.v1.set_random_seed(1)
    except:
        tf.random.set_seed(1)

    model = keras.Sequential([
      keras.layers.Flatten(input_shape=(28, 28)),
      keras.layers.Dense(128, activation='relu'),
      keras.layers.Dropout(rate = dropout),
      keras.layers.Dense(10, activation='softmax')
    ])

    adam = keras.optimizers.Adam(lr=learningrate)
    Adagrad = keras.optimizers.Adagrad(lr=learningrate)
    RMSprop = keras.optimizers.RMSprop(lr=learningrate)
    SGD = keras.optimizers.SGD(lr=learningrate)
    adadelta = keras.optimizers.Adadelta(learning_rate=learningrate)
    adammax = keras.optimizers.Adamax(learning_rate=learningrate)
    nadam = keras.optimizers.Nadam(learning_rate=learningrate)
    ftrl = keras.optimizers.Ftrl(learning_rate=learningrate)
    optimizerarray = [adam, SGD, RMSprop, Adagrad, adadelta,adammax,nadam,ftrl]

    if round(optimizer) < -0.5:
        optimizer = 0
    elif round(optimizer) > 7.5:
        optimizer = 7

    model.compile(optimizer=optimizerarray[round(optimizer)],
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
    return model

def create_model_medium(optimizer,learningrate, dropout):
    try:
        tf.compat.v1.set_random_seed(1)
    except:
        tf.random.set_seed(1)

    model = keras.models.Sequential([
            keras.layers.Flatten(input_shape=(28,28)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(rate = dropout),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(rate = dropout),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(rate = dropout),
            keras.layers.Dense(10, activation='softmax')
            ])

    adam = keras.optimizers.Adam(lr=learningrate)
    Adagrad = keras.optimizers.Adagrad(lr=learningrate)
    RMSprop = keras.optimizers.RMSprop(lr=learningrate)
    SGD = keras.optimizers.SGD(lr=learningrate)
    adadelta = keras.optimizers.Adadelta(learning_rate=learningrate)
    adammax = keras.optimizers.Adamax(learning_rate=learningrate)
    nadam = keras.optimizers.Nadam(learning_rate=learningrate)
    ftrl = keras.optimizers.Ftrl(learning_rate=learningrate)
    optimizerarray = [adam, SGD, RMSprop, Adagrad, adadelta,adammax,nadam,ftrl]

    if round(optimizer) < -0.5:
        optimizer = 0
    elif round(optimizer) > 7.5:
        optimizer = 7

    model.compile(optimizer=optimizerarray[round(optimizer)],
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

    return model

def create_model_big(optimizer,learningrate, dropout):
    try:
        tf.compat.v1.set_random_seed(1)
    except:
        tf.random.set_seed(1)

    model = keras.models.Sequential([
            keras.layers.Flatten(input_shape=(28,28)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(rate = dropout),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(rate = dropout),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(rate = dropout),
            keras.layers.Dense(10, activation='softmax')
            ])

    adam = keras.optimizers.Adam(lr=learningrate)
    Adagrad = keras.optimizers.Adagrad(lr=learningrate)
    RMSprop = keras.optimizers.RMSprop(lr=learningrate)
    SGD = keras.optimizers.SGD(lr=learningrate)
    adadelta = keras.optimizers.Adadelta(learning_rate=learningrate)
    adammax = keras.optimizers.Adamax(learning_rate=learningrate)
    nadam = keras.optimizers.Nadam(learning_rate=learningrate)
    ftrl = keras.optimizers.Ftrl(learning_rate=learningrate)
    
    optimizerarray = [adam, SGD, RMSprop, Adagrad, adadelta,adammax,nadam,ftrl]

    if round(optimizer) < -0.5:
        optimizer = 0
    elif round(optimizer) > 7.5:
        optimizer = 7
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizerarray[optimizer], metrics=['accuracy'])
    return model

    
def create_model_cnn(optimizer,learningrate, dropout):
    try:
        tf.compat.v1.set_random_seed(1)
    except:
        tf.random.set_seed(1)

    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32, (3, 3), padding='same',input_shape=[32,32,3]))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv2D(32, (3, 3)))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(rate = dropout))

    model.add(keras.layers.Conv2D(64, (3, 3), padding='same'))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv2D(64, (3, 3)))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(rate = dropout))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dropout(rate = dropout))
    model.add(keras.layers.Dense(10))
    model.add(keras.layers.Activation('softmax')) 

    #%%si
    ### Optimizer

    adam = keras.optimizers.Adam(lr=learningrate)
    Adagrad = keras.optimizers.Adagrad(lr=learningrate)
    RMSprop = keras.optimizers.RMSprop(lr=learningrate)
    SGD = keras.optimizers.SGD(lr=learningrate)
    adadelta = keras.optimizers.Adadelta(learning_rate=learningrate)
    adammax = keras.optimizers.Adamax(learning_rate=learningrate)
    nadam = keras.optimizers.Nadam(learning_rate=learningrate)
    ftrl = keras.optimizers.Ftrl(learning_rate=learningrate)
    optimizerarray = [adam, SGD, RMSprop, Adagrad, adadelta,adammax,nadam,ftrl]


    model.compile(loss='categorical_crossentropy',
              optimizer=optimizerarray[round(optimizer)],
              metrics=['accuracy'])
    return model

def re_acc(tree):
    return tree["acc"]

def save_params(save_file, means, params, datenset, iterations, knn_size,small_dataset,algorithmus,acc):
        try:
            with open(self.save_file, "r") as f:
                data = json.load(f)
        except:
            data = {}
        #self.individuals = list(sorted(param, key=lambda x: x.var_acc, reverse=True))
        pop = []
        family_tree = {}
        print(means)
        print(params)
        for mean, param in zip(means, params):
            param["acc"] = mean
            pop.append(param)
        pop = list(sorted(pop, key=re_acc ,reverse=True))
        i = 0
        for param in pop:
            if i == 0:
                write_cell(path_to_file = "../data/evaluation.xlsx",dataset = datenset ,iterations = iterations,
                knn_size = knn_size,small_dataset =small_dataset,algorithmus= algorithmus,acc = acc)
            generation = {
                "name": i,
                "learningrate": str(param["learningrate"]),
                "dropout": str(param["dropout"]),
                "epoch": str(param["epochs"]),
                "batchsize": str(param["batch_size"]),
                "optimizer": str(param["optimizer"]),
                "acc": str(param["acc"]),
                "loss": str(0)
            }
            family_tree[i] = generation 
            i += 1
        del i
        data.update(family_tree)
        with open(save_file, "w") as outfile:
            json.dump(data, outfile, indent=2)
        print("saved winnerpopulation gens into ", save_file)


def save_init_data(save_file,dataset, iteration, knn_size,small_dataset,algorithmus,multiprocessing,gpu):
        try:
            with open(save_file, "r") as f:
                data = json.load(f)
        except:
            data = {}

        configurations = { "config": {
            
            "dataset" :dataset,
            "knn_size" : knn_size,
            "small_dataset" : small_dataset,
            "iteration" : iteration,
            "PC_name":  socket.gethostname(),
            "Multiprocessing" : multiprocessing,
            "gpu" : gpu}
        }
        generation = {"generation":{}}
        round_time = {"round_time":{}}
        data.update(configurations)
        data.update(generation)
        data.update(round_time)
        with open(save_file, "w") as outfile:
            json.dump(data, outfile, indent=2)
        del configurations
        del generation