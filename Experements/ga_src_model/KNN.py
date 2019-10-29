from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import gc
import time
import numpy as np
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

def train_and_evalu_model_mean(Neuronen_Layer1,Neuronen_Layer2,Neuronen_Layer3):
    var_learningrate = 0.05
    var_dropout = 0.25
    var_epoch = 20
    var_batch_size = 32
    optimizer = 3
    mean_var = 3 
    #%%
    ### Daten
    print("var_learningrate", var_learningrate, "var_dropout", var_dropout, "var_epoch", var_epoch, "var_batch_size", var_batch_size)
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
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
      keras.layers.Dense(Neuronen_Layer1, activation='relu'),
      keras.layers.Dropout(var_dropout),
      keras.layers.Dense(Neuronen_Layer2, activation='relu'),
      keras.layers.Dropout(var_dropout),
      keras.layers.Dense(Neuronen_Layer3, activation='relu'),
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
    test_loss_all = 0
    test_acc_all = 0
    t_eval_all = 0
    variables_all = 0
    for i in range(0,mean_var):
        ### Model fit
        #model.fit(small_train_images, small_train_labels, epochs=int(var_epoch),batch_size=int(var_batch_size),use_multiprocessing=True, workers=2)
        model.fit(train_images, train_labels, epochs=int(var_epoch),batch_size=int(var_batch_size),use_multiprocessing=True, workers=2)
        #%%
        ### Model evalu

        #test_loss, test_acc = model.evaluate(small_test_images, small_test_labels)
        start = time.time()
        test_loss, test_acc = model.evaluate(test_images, test_labels)
        t_eval = time.time() - start
        variables = np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()])
        t_eval_all += t_eval
        test_loss_all += test_loss
        test_acc_all += test_acc 
        variables_all += variables
    t_eval_mean = t_eval_all / mean_var
    test_loss_mean = test_loss_all / mean_var
    test_acc_mean = test_acc / mean_var
    variables_mean = variables_all / mean_var
    print("test_loss: ",test_loss , "test_acc: ", test_acc)
    gc.collect()

    return test_loss_mean, test_acc_mean , t_eval_mean, variables


