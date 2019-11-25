# coding=utf-8

import os
import time
import datetime
import sys
import argparse

import sklearn
import numpy as np
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import tensorflow as tf
from tensorflow import keras

from tools import gs_tools
from tools.KNN import train_and_evalu
from tools import plotting

def main(dataset_arg="mnist_fashion", knn_size_arg="small", iteration=50,
         gpu_arg=False, multiprocessing_arg=2, small_dataset_arg=False):
    algorithmus = "RS"
    start_time = time.time()
    iteration = iteration
    gpu = gpu_arg
    dataset = dataset_arg
    small_dataset = small_dataset_arg
    knn_size = knn_size_arg
    multiprocessing_flag = True
    multiprocessing = multiprocessing_arg
    show_plot = False

    if multiprocessing == 0:
        multiprocessing_flag = False

    if gpu == True:
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        config.gpu_options.allow_growth = False
        session = tf.compat.v1.Session(config=config)
        keras.backend.set_session(session)
    else:
        ##zwingen auf cpu
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    dir_path = os.path.dirname(os.path.abspath(__file__))
    save_file = os.path.join(dir_path,
                             "../data/{}.{}.{}_rs.json".format(datetime.datetime.now().year,
                                                               datetime.datetime.now().month,
                                                               datetime.datetime.now().day))
    if os.path.isfile(save_file):
        for i in range(1, 10):
            save_file = os.path.join(dir_path,
                                     "../data/{}.{}.{}-{}_rs.json"
                                     .format(datetime.datetime.now().year,
                                             datetime.datetime.now().month,
                                             datetime.datetime.now().day, i))
            if os.path.isfile(save_file) == False:
                break

    gs_tools.save_init_data(save_file=save_file, dataset=dataset, iteration=iteration,
                         knn_size=knn_size, small_dataset=small_dataset, algorithmus=algorithmus,
                         multiprocessing=multiprocessing, gpu=gpu)

    fully = True
    cnn = False
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
    else:
        print("Wrong Dataset ")

    train_images = train_images / 255.0
    test_images = test_images / 255.0

     #extra feature um Datenset künstlich zu verkleinern
    small_train_images, small_test_images, small_train_labels, small_test_labels = sklearn.model_selection.train_test_split(
        train_images, train_labels, test_size=0.9, shuffle=False)
    if fully:
        if knn_size == "small":
            model = KerasClassifier(build_fn=gs_tools.create_model_small, verbose=0,
                                    use_multiprocessing=multiprocessing_flag, workers=multiprocessing)
        elif knn_size == "medium":
            model = KerasClassifier(build_fn=gs_tools.create_model_medium, verbose=0,
                                    use_multiprocessing=multiprocessing_flag, workers=multiprocessing)
        elif knn_size == "medium":
            model = KerasClassifier(build_fn=gs_tools.create_model_big, verbose=0,
                                    use_multiprocessing=multiprocessing_flag, workers=multiprocessing)
    elif cnn:
        model = KerasClassifier(build_fn=gs_tools.create_model_cnn, verbose=0,
                                use_multiprocessing=multiprocessing_flag, workers=multiprocessing)

    optimizers = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    epochs = np.array([10, 20, 30, 40, 50, 60, 70, 80])
    batchsize = np.array([8, 16, 32, 40, 48, 56, 64, 72])
    learningrate = np.array([0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1])
    dropout = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

    param_grid = dict(optimizer=optimizers, epochs=epochs,
                      batch_size=batchsize, learningrate=learningrate, dropout=dropout)

    random_search = sklearn.model_selection.RandomizedSearchCV(estimator=model,
                                                               param_distributions=param_grid,
                                                               n_iter=iteration, n_jobs=multiprocessing)
    if small_dataset:
        random_search.fit(small_train_images, small_train_labels,
                          use_multiprocessing=multiprocessing_flag, workers=multiprocessing)
    else:
        random_search.fit(train_images, train_labels, use_multiprocessing=True, workers=2)

    print("Best: %f using %s" % (random_search.best_score_, random_search.best_params_))
    means = random_search.cv_results_['mean_test_score']
    stds = random_search.cv_results_['std_test_score']
    params = random_search.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    end_time = time.time()
    print("finisched in ", end_time-start_time)
    all_time = end_time-start_time
    best = random_search.best_params_


    optimizer = best["optimizer"]
    epochs = best["epochs"]
    batchsize = best["batch_size"]
    learningrate = best["learningrate"]
    dropout = best["dropout"]
    gene = [learningrate, dropout, epochs, batchsize, optimizer]
    test_loss, test_acc, variables, precision_score_var, recall_score_var, f1_score_var, cm = train_and_evalu(gene=gene, dataset=dataset, knn_size=knn_size, small_dataset=small_dataset, gpu=gpu, f1=True)
    gs_tools.save_params(dir_path, save_file,all_time, means, params, dataset, iteration,
                      knn_size, small_dataset, algorithmus, test_acc, 
                      precision_score_var, recall_score_var, f1_score_var, cm)



if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Grid/Random Search")
    PARSER.add_argument('dataset', help='which dataset: mnist_fashion, mnist_digits, cifar10 ', type=str, default="mnist_fashion")
    PARSER.add_argument("knn_size", help='which size of KNN: small, big ', type=str, default="small")
    PARSER.add_argument("iteration", help='iteration ', type=int, default=50)   
    PARSER.add_argument("gpu", help='if gpu enable True or False ', type=bool, default=False)
    PARSER.add_argument("multiprocessing", help='How many parallel processes: 2, 4, if 0 then no multiprocess', type=int, default=2)
    PARSER.add_argument("small_dataset", help="to train with only 0.1 of the original Dataset", type=bool, default=False)
    ARGS = PARSER.parse_args()

    main(dataset_arg=ARGS.dataset, knn_size_arg=ARGS.knn_size, iteration=ARGS.iteration, 
         gpu_arg=ARGS.gpu, multiprocessing_arg=ARGS.multiprocessing, small_dataset_arg=ARGS.small_dataset)
