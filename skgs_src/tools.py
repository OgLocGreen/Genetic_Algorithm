from tensorflow import keras
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import json
import tensorflow as tf


def create_model(optimizer,learningrate, dropout):
    try:
        tf.set_random_seed(1)
    except:
        tf.random.set_seed(1)
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

def re_acc(tree):
    return tree["acc"]

def save_params(save_file, means, params):
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

