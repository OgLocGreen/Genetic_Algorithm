# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import random
import multiprocessing
import json
import gc
import datetime
import time
import os

import plotting
import KNN
import crossover
import mutation
import individual
import population

import argparse 
import sys


def main(dataset_arg= "mnist_fashion", knn_size_arg = "small" ,pop_size_arg = 10, generations_arg = 2,gpu_arg = False, multiprocessing_arg = 2):
    pop_size = pop_size_arg
    mutate_prob = 0.1 #sigma for random.gauss()
    retain = 0.8
    random_retain = 0.05
    GENERATIONS = generations_arg
    gpu = gpu_arg
    dataset = dataset_arg
    small_dataset = False
    knn_size = knn_size_arg

    multiprocessing_flag = True
    multiprocessing_var = multiprocessing_arg
    SHOW_PLOT = False

    if multiprocessing == 0:
        multiprocessing_flag = False

    pop = population.Population(pop_size=pop_size, mutate_prob=mutate_prob, retain=retain, random_retain=random_retain,
                                        generations=GENERATIONS,dataset=dataset,small_dataset=small_dataset,knn_size=knn_size, gpu=gpu , multiprocessing= multiprocessing_var)
    start = time.time()
    lastround = start
    round_time = []
    pop.save_init_data()
    for x in range(GENERATIONS):
        pop.grade(generation=x)
        if pop.done:
            end = time.time()
            print("Finished at generation:", x, ", Population fistness:", pop.fitness_history[-1])
            break
        else:
            pop.evolve()
            print("Finished with ",x,"Generation" )
        newround = time.time()
        round_time.append(newround - lastround)
        lastround = newround
        gc.collect()

    end_2 = time.time()
    pop.save_gens_winner()
    for i in range(0,len(round_time)):
        print("Round: ", i," Time: ", round_time[i])
    pop.save_end_data(round_time)

#%%
    # Plot fitness history
    if SHOW_PLOT:
        print("Showing fitness history graph")
        plt.plot(np.arange(len(pop.fitness_history)), pop.fitness_history)
        plt.ylabel('Fitness')
        plt.xlabel('Generations')
        plt.title('Fitness - pop_size {} mutate_prob {} retain {} random_retain {}'.format(pop_size, mutate_prob, retain, random_retain))
        plt.show()

        plotting.scatterplot(pop.save_file)
        plotting.joint_plot(pop.save_file)
    print("FINISCHED!!! after ", end_2-start, "seconds")
    print(round_time)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Genetic Algorithm")
    parser.add_argument('dataset', help='which dataset: mnist_fashion, mnist_digits, cifar10 ' , type=str, default= "mnist_fashion")
    parser.add_argument("knn_size", help='which size of KNN: small, medium, big ' ,type=str, default= "small")
    parser.add_argument("pop_size", help='Populationsize ',type=int, default= 10)
    parser.add_argument("generations", help='how many Generations ',type=int, default= 3)    
    parser.add_argument("gpu", help='if gpu enable True or False ',type=bool, default= False)
    parser.add_argument("multiprocessing", help='How many parallel processes: 2, 4, if 0 then no multiprocess',type=int, default= 2)
    args = parser.parse_args()

    main(dataset_arg = args.dataset, knn_size_arg= args.knn_size, pop_size_arg= args.pop_size, generations_arg=args.generations, 
        	gpu_arg= args.gpu, multiprocessing_arg = args.multiprocessing)

    #main(args.dataset, args.knn_size, args.pop_size,
    #                 args.generations, args.gpu, args.multiprocessing)
    
    
    


