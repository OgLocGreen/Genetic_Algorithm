from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import gc
import numpy as np 


# GPU auf maximal 30% Leistung
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

def train_and_evalu(gene,dataset,knn_size = "small" ,small_dataset = False ,gpu = False):

    if gpu == True:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    ### Daten   
    var_learningrate,var_dropout,var_epoch,var_batch_size,optimizer = gene[0],gene[1],gene[2],gene[3],gene[4]

    print("var_learningrate", var_learningrate, "var_dropout", var_dropout, "var_epoch", var_epoch, "var_batch_size", var_batch_size, "optimizer", optimizer)
    fully = False
    cnn = False

    if dataset == "mnist_fashion": 
        (train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
        fully = True
    elif dataset == "minst_digits":
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
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(var_dropout),
            keras.layers.Dense(10, activation='softmax')
            ])
        elif knn_size == "big":
            model = keras.models.Sequential([
            keras.layers.Flatten(input_shape=train_images.shape[1:]),
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


    #%%si
    ### Optimizer

    adam = keras.optimizers.Adam(lr=var_learningrate)
    Adagrad = keras.optimizers.Adagrad(lr=var_learningrate)
    RMSprop = keras.optimizers.RMSprop(lr=var_learningrate)
    SGD = keras.optimizers.SGD(lr=var_learningrate)


    optimizerarray = [adam, SGD, RMSprop, Adagrad]


    if round(optimizer) < -0.5:
        optimizer = 0
    elif round(optimizer) > 3.5:
        optimizer = 3

    if fully == True:
        model.compile(optimizer=optimizerarray[round(optimizer)],
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
    elif cnn == True:
        model.compile(loss='categorical_crossentropy',
              optimizer=optimizerarray[round(optimizer)],
              metrics=['accuracy'])
    #%%
    ### Tensorboard

    #tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

    #%%
    ### Model fit and Evaluate
    if small_dataset is True:
       model.fit(small_train_images, small_train_labels, epochs=int(var_epoch),batch_size=int(var_batch_size),use_multiprocessing=True, workers=2,verbose=0)
       test_loss, test_acc = model.evaluate(small_test_images, small_test_labels,verbose=0)
    else:
        model.fit(train_images, train_labels, epochs=int(var_epoch),batch_size=int(var_batch_size),use_multiprocessing=True, workers=2,verbose=0)
        test_loss, test_acc = model.evaluate(test_images, test_labels,verbose=0)
    variables = np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()])
    print("test_loss: ",test_loss , "test_acc: ", test_acc, "variables",variables)
    gc.collect()
    return test_loss, test_acc, variables


def train_and_evalu_mnist_fashion(var_learningrate,var_dropout,var_epoch,var_batch_size,optimizer):
    ### Daten
    print("var_learningrate", var_learningrate, "var_dropout", var_dropout, "var_epoch", var_epoch, "var_batch_size", var_batch_size, "optimizer", optimizer)
    
    
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0


    #extra feature um Datenset künstlich zu verkleinern
    small_train_images, small_test_images, small_train_labels, small_test_labels = train_test_split(
        train_images, train_labels, test_size=0.9, shuffle=False)

    #%%
    ### Model
    try:
        tf.set_random_seed(1)
    except:
        tf.random.set_seed(1)
    

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


    optimizerarray = [adam, SGD, RMSprop, Adagrad]


    if round(optimizer) < -0.5:
        optimizer = 0
    elif round(optimizer) > 3.5:
        optimizer = 3

    model.compile(optimizer=optimizerarray[round(optimizer)],
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    #%%
    ### Tensorboard

    #tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

    #%%
    ### Model fit and Evaluate
    if small_dataset is True:
       model.fit(small_train_images, small_train_labels, epochs=int(var_epoch),batch_size=int(var_batch_size),use_multiprocessing=True, workers=2,verbose=0)
       test_loss, test_acc = model.evaluate(small_test_images, small_test_labels)
    else:
        model.fit(train_images, train_labels, epochs=int(var_epoch),batch_size=int(var_batch_size),use_multiprocessing=True, workers=2,verbose=0)
        test_loss, test_acc = model.evaluate(test_images, test_labels)
    variables = np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()])
    print("test_loss: ",test_loss , "test_acc: ", test_acc, "variables",variables)
    gc.collect()
    return test_loss, test_acc, variables


def train_and_evalu_cifar10(var_learningrate,var_dropout,var_epoch,var_batch_size,optimizer):
    small_dataset = False
    
    #%%
    ### Daten
    print("var_learningrate", var_learningrate, "var_dropout", var_dropout, "var_epoch", var_epoch, "var_batch_size", var_batch_size, "optimizer", optimizer)
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0


    #extra feature um Datenset künstlich zu verkleinern
    small_train_images, small_test_images, small_train_labels, small_test_labels = train_test_split(
        train_images, train_labels, test_size=0.9, shuffle=False)

    #%%
    ### Model
    try:
        tf.set_random_seed(1)
    except:
        tf.random.set_seed(1)
    model = keras.models.Sequential([
      keras.layers.Flatten(input_shape=(32, 32,3 )),
      keras.layers.Dense(128, activation='relu'),
      keras.layers.Dropout(var_dropout),
      keras.layers.Dense(256, activation='relu'),
      keras.layers.Dropout(var_dropout),
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


    optimizerarray = [adam, SGD, RMSprop, Adagrad]


    if round(optimizer) < -0.5:
        optimizer = 0
    elif round(optimizer) > 3.5:
        optimizer = 3

    model.compile(optimizer=optimizerarray[round(optimizer)],
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    #%%
    ### Tensorboard

    #tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

    #%%
    ### Model fit and Evaluate
       ### Model fit and Evaluat
    test_loss_all = 0
    test_acc_all = 0
    variables_all = 0
    mean_var = 3
    mean = True
    if(mean == 1):
        for i in range(0,mean_var):
            if small_dataset is True:
                model.fit(small_train_images, small_train_labels, epochs=int(var_epoch),batch_size=int(var_batch_size),use_multiprocessing=True, workers=2,verbose=0)
                test_loss, test_acc = model.evaluate(small_test_images, small_test_labels)
                variables = np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()])
            else:
                model.fit(train_images, train_labels, epochs=int(var_epoch),batch_size=int(var_batch_size),use_multiprocessing=True, workers=2,verbose=0)
                test_loss, test_acc = model.evaluate(test_images, test_labels)
                variables = np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()])
            test_loss_all += test_loss
            test_acc_all += test_acc
            variables_all += variables 
        test_loss_mean = test_loss_all / mean_var
        test_acc_mean = test_acc_all /mean_var
        variables_mean = variables_all /mean_var
    print ("variables: ",variables)
    print("test_loss: ",test_loss , "test_acc: ", test_acc)
    gc.collect()
    return test_loss_mean, test_acc_mean , variables_mean

"""
def train_and_evalu_cifar10_mean(var_learningrate,var_dropout,var_epoch,var_batch_size,optimizer):

    small_dataset = False

    ##  angaben für Mean nacht als argumente übergeben
    mean = True
    mean_var = 3
    
    #%%
    ### Daten
    print("var_learningrate", var_learningrate, "var_dropout", var_dropout, "var_epoch", var_epoch, "var_batch_size", var_batch_size)
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0


    #extra feature um Datenset künstlich zu verkleinern
    small_train_images, small_test_images, small_train_labels, small_test_labels = train_test_split(
        train_images, train_labels, test_size=0.9, shuffle=False)

    #%%
    ### Model
    try:
        tf.set_random_seed(1)
    except:
        tf.random.set_seed(1)
    model = keras.models.Sequential([
      keras.layers.Flatten(input_shape=(32, 32, 3)),
      keras.layers.Dense(128, activation='relu'),
      keras.layers.Dropout(var_dropout),
      keras.layers.Dense(256, activation='relu'),
      keras.layers.Dropout(var_dropout),
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


    optimizerarray = [adam, SGD, RMSprop, Adagrad]


    if round(optimizer) < -0.5:
        optimizer = 0
    elif round(optimizer) > 3.5:
        optimizer = 3

    model.compile(optimizer=optimizerarray[round(optimizer)],
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    #%%
    ### Tensorboard

    #tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

    #%%
    ### Model fit and Evaluat
    test_loss_all = 0
    test_acc_all = 0
    variables_all = 0
    if(mean == 1):
        for i in range(0,mean_var):
            if small_dataset is True:
                model.fit(small_train_images, small_train_labels, epochs=int(var_epoch),batch_size=int(var_batch_size),use_multiprocessing=True, workers=2,verbose=0)
                test_loss, test_acc = model.evaluate(small_test_images, small_test_labels)
                variables = np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()])
            else:
                model.fit(train_images, train_labels, epochs=int(var_epoch),batch_size=int(var_batch_size),use_multiprocessing=True, workers=2,verbose=0)
                test_loss, test_acc = model.evaluate(test_images, test_labels)
                variables = np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()])
            test_loss_all += test_loss
            test_acc_all += test_acc
            variables_all += variables 
        test_loss_mean = test_loss_all / mean_var
        test_acc_mean = test_acc_all /mean_var
        variables_mean = variables_all /mean_var
    else: ## wird eh nicht gebraucht
        if small_dataset is True:
            model.fit(small_train_images, small_train_labels, epochs=int(var_epoch),batch_size=int(var_batch_size),use_multiprocessing=True, workers=2,verbose=0)
            test_loss, test_acc = model.evaluate(small_test_images, small_test_labels)
        else:
            model.fit(train_images, train_labels, epochs=int(var_epoch),batch_size=int(var_batch_size),use_multiprocessing=True, workers=2,verbose=0)
            test_loss, test_acc = model.evaluate(test_images, test_labels)

    print("test_loss: ",test_loss , "test_acc: ", test_acc,"variabkes_mean: ",variables_mean)
    gc.collect()
    return test_loss_mean, test_acc_mean, variables_mean
"""

def train_and_evalu_cifar100(var_learningrate,var_dropout,var_epoch,var_batch_size,optimizer):
    small_dataset = False
    
    #%%
    ### Daten
    print("var_learningrate", var_learningrate, "var_dropout", var_dropout, "var_epoch", var_epoch, "var_batch_size", var_batch_size)
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar100.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0


    #extra feature um Datenset künstlich zu verkleinern
    small_train_images, small_test_images, small_train_labels, small_test_labels = train_test_split(
        train_images, train_labels, test_size=0.9, shuffle=False)

    #%%
    ### Model
    try:
        tf.set_random_seed(1)
    except:
        tf.random.set_seed(1)
    model = keras.models.Sequential([
      keras.layers.Flatten(input_shape=(32, 32,3 )),
      keras.layers.Dense(128, activation='relu'),
      keras.layers.Dropout(var_dropout),
      keras.layers.Dense(256, activation='relu'),
      keras.layers.Dropout(var_dropout),
      keras.layers.Dense(128, activation='relu'),
      keras.layers.Dropout(var_dropout),
      keras.layers.Dense(100, activation='softmax')
    ])


    #%%si
    ### Optimizer




    adam = keras.optimizers.Adam(lr=var_learningrate)
    Adagrad = keras.optimizers.Adagrad(lr=var_learningrate)
    RMSprop = keras.optimizers.RMSprop(lr=var_learningrate)
    SGD = keras.optimizers.SGD(lr=var_learningrate)


    optimizerarray = [adam, SGD, RMSprop, Adagrad]


    if round(optimizer) < -0.5:
        optimizer = 0
    elif round(optimizer) > 3.5:
        optimizer = 3

    model.compile(optimizer=optimizerarray[round(optimizer)],
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    #%%
    ### Tensorboard

    #tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

    #%%
    ### Model fit and Evaluate
       ### Model fit and Evaluat
    test_loss_all = 0
    test_acc_all = 0
    variables_all = 0
    mean_var = 3
    mean = True
    if(mean == 1):
        for i in range(0,mean_var):
            if small_dataset is True:
                model.fit(small_train_images, small_train_labels, epochs=int(var_epoch),batch_size=int(var_batch_size),use_multiprocessing=True, workers=2,verbose=0)
                test_loss, test_acc = model.evaluate(small_test_images, small_test_labels)
                variables = np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()])
            else:
                model.fit(train_images, train_labels, epochs=int(var_epoch),batch_size=int(var_batch_size),use_multiprocessing=True, workers=2,verbose=0)
                test_loss, test_acc = model.evaluate(test_images, test_labels)
                variables = np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()])
            test_loss_all += test_loss
            test_acc_all += test_acc
            variables_all += variables 
        test_loss_mean = test_loss_all / mean_var
        test_acc_mean = test_acc_all /mean_var
        variables_mean = variables_all /mean_var
    print ("variables: ",variables)
    print("test_loss: ",test_loss , "test_acc: ", test_acc)
    gc.collect()
    return test_loss_mean, test_acc_mean , variables_mean


"""
def train_and_evalu_cifar100_mean(var_learningrate,var_dropout,var_epoch,var_batch_size,optimizer):

    small_dataset = False

    ##  angaben für Mean nacht als argumente übergeben
    mean = True
    mean_var = 3
    
    #%%
    ### Daten
    print("var_learningrate", var_learningrate, "var_dropout", var_dropout, "var_epoch", var_epoch, "var_batch_size", var_batch_size)
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar100.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0


    #extra feature um Datenset künstlich zu verkleinern
    small_train_images, small_test_images, small_train_labels, small_test_labels = train_test_split(
        train_images, train_labels, test_size=0.9, shuffle=False)

    #%%
    ### Model
    tf.set_random_seed(1)
    model = keras.models.Sequential([
      keras.layers.Flatten(input_shape=(32, 32, 3)),
      keras.layers.Dense(128, activation='relu'),
      keras.layers.Dropout(var_dropout),
      keras.layers.Dense(256, activation='relu'),
      keras.layers.Dropout(var_dropout),
      keras.layers.Dense(128, activation='relu'),
      keras.layers.Dropout(var_dropout),
      keras.layers.Dense(100, activation='softmax')
    ])


    #%%si
    ### Optimizer




    adam = keras.optimizers.Adam(lr=var_learningrate)
    Adagrad = keras.optimizers.Adagrad(lr=var_learningrate)
    RMSprop = keras.optimizers.RMSprop(lr=var_learningrate)
    SGD = keras.optimizers.SGD(lr=var_learningrate)


    optimizerarray = [adam, SGD, RMSprop, Adagrad]


    if round(optimizer) < -0.5:
        optimizer = 0
    elif round(optimizer) > 3.5:
        optimizer = 3

    model.compile(optimizer=optimizerarray[round(optimizer)],
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    #%%
    ### Tensorboard

    #tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

    #%%
    ### Model fit and Evaluate
    test_loss_all = 0
    test_acc_all = 0
    if(mean == 1):
        for i in range(0,mean_var):
            if small_dataset is True:
                model.fit(small_train_images, small_train_labels, epochs=int(var_epoch),batch_size=int(var_batch_size),use_multiprocessing=True, workers=2, verbose=0)
                test_loss, test_acc = model.evaluate(small_test_images, small_test_labels)
            else:
                model.fit(train_images, train_labels, epochs=int(var_epoch),batch_size=int(var_batch_size),use_multiprocessing=True, workers=2, verbose=0)
                test_loss, test_acc = model.evaluate(test_images, test_labels)
            test_loss_all += test_loss
            test_acc_all += test_acc 
        test_loss_mean = test_loss_all / mean_var
        test_acc_mean = test_acc /mean_var
    else: ## wird eh nicht gebraucht
        if small_dataset is True:
            model.fit(small_train_images, small_train_labels, epochs=int(var_epoch),batch_size=int(var_batch_size),use_multiprocessing=True, workers=2, verbose=0)
            test_loss, test_acc = model.evaluate(small_test_images, small_test_labels)
        else:
            model.fit(train_images, train_labels, epochs=int(var_epoch),batch_size=int(var_batch_size),use_multiprocessing=True, workers=2, verbose=0)
            test_loss, test_acc = model.evaluate(test_images, test_labels)

    print("test_loss: ",test_loss , "test_acc: ", test_acc)
    gc.collect()
    return test_loss_mean, test_acc_mean
"""


if __name__ == "__main__":
    data = train_and_evalu_cifar10(var_learningrate = 0.05,var_dropout=0.25,var_epoch=100,var_batch_size=16,optimizer=3)
    print(data)