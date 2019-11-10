# coding=utf-8
import numpy as np
import os
from tensorflow import keras
from time import time
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split
import sklearn
import time
import json
import datetime
import socket
import sys
import argparse

import plotting
import tools
sys.path.append('../')
from src_evaluation.evaluation import write_cell
from ga_src_hyper.KNN import train_and_evalu




def main(dataset_arg= "mnist_fashion", knn_size_arg = "small" ,iteration = 50 ,gpu_arg = False, multiprocessing_arg = 2):
    algorithmus = "RS"
    start_time = time.time()
    iteration = iteration
    gpu = gpu_arg
    dataset = dataset_arg
    small_dataset = False
    knn_size = knn_size_arg

    multiprocessing_flag = True
    multiprocessing = multiprocessing_arg
    SHOW_PLOT = False

    if multiprocessing == 0:
        multiprocessing_flag = False
    if gpu ==False:
        ##zwingen auf cpu
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        pass

    

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

    tools.save_init_data(save_file=save_file, dataset=dataset, iteration=iteration, knn_size=knn_size,small_dataset=small_dataset,algorithmus=algorithmus,multiprocessing=multiprocessing,gpu=gpu)

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
    small_train_images, small_test_images, small_train_labels, small_test_labels = train_test_split(
        train_images, train_labels, test_size=0.9, shuffle=False)
    if fully:
        if knn_size == "small":
            model = KerasClassifier(build_fn=tools.create_model_small,verbose=0,use_multiprocessing=multiprocessing_flag, workers=multiprocessing)
        elif knn_size == "medium":
            model = KerasClassifier(build_fn=tools.create_model_medium,verbose=0,use_multiprocessing=multiprocessing_flag, workers=multiprocessing)
        elif knn_size == "medium":
            model = KerasClassifier(build_fn=tools.create_model_big,verbose=0,use_multiprocessing=multiprocessing_flag, workers=multiprocessing)
    elif cnn:
        model = KerasClassifier(build_fn=tools.create_model_cnn,verbose=0,use_multiprocessing=multiprocessing_flag, workers=multiprocessing)

    optimizers = np.array([0, 1, 2, 3, 4 ,5 ,6 ,7 ])
    epochs = np.array([30,40,50, 60, 70, 80,90, 100])
    batchsize = np.array([8,16,32,40,48,56, 64,72])
    learningrate = np.array([0.00005, 0.0001 ,0.0005, 0.001, 0.005, 0.01, 0.05, 0.1])
    dropout = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6,0.7])

    param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batchsize, learningrate=learningrate, dropout=dropout)

    random_search = sklearn.model_selection.RandomizedSearchCV(estimator=model,
                                    param_distributions=param_grid,
                                    n_iter=iteration, n_jobs=multiprocessing)
                                    
    if small_dataset:
        random_search.fit(small_train_images, small_train_labels,use_multiprocessing=multiprocessing_flag, workers=multiprocessing)
    else:
        random_search.fit(train_images, train_labels,use_multiprocessing=False, workers=1)  ##Momentanes Problem

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
    gene=[learningrate,dropout,epochs,batchsize,optimizer]

    test_loss, test_acc, variables =  train_and_evalu(gene=gene,dataset = dataset, knn_size= knn_size, small_dataset = small_dataset, gpu = gpu)

    tools.save_params(save_file, means, params,dataset, iteration, knn_size,small_dataset,algorithmus,test_acc)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Grid/Random Search")
    parser.add_argument('dataset', help='which dataset: mnist_fashion, mnist_digits, cifar10 ' , type=str, default = "mnist_fashion" )
    parser.add_argument("knn_size", help='which size of KNN: small, medium, big ' ,type=str, default = "small")
    parser.add_argument("iteration", help='iteration ',type=int, default = 50)   
    parser.add_argument("gpu", help='if gpu enable True or False ',type=bool, default = False)
    parser.add_argument("multiprocessing", help='How many parallel processes: 2, 4, if 0 then no multiprocess',type=int, default = 2 )
    args = parser.parse_args()

    main(dataset_arg = args.dataset, knn_size_arg= args.knn_size,  iteration=args.iteration, 
        	gpu_arg= args.gpu, multiprocessing_arg = args.multiprocessing)
