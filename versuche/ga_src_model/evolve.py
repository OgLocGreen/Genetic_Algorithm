import numpy as np
import matplotlib.pyplot as plt
import random
import multiprocessing
import json
import gc
import datetime
import time
import os

from plotting import plot_winner, plot_all, plot_histogram_all, scatterplot
import KNN
import crossover
import mutation
import individual
import population






if __name__ == "__main__":
    pop_size = 10
    mutate_prob = 0.1 #sigma for random.gauss()
    retain = 0.8
    random_retain = 0.05
    GENERATIONS = 5

    multiprocessing_flag = False
    multiprocessing_var = 2

    SHOW_PLOT = True

    pop = population.Population(pop_size=pop_size, mutate_prob=mutate_prob, retain=retain, random_retain=random_retain)
    start = time.time()
    round_time = []
    for x in range(GENERATIONS):
        if multiprocessing_flag:
            pop.grade_multi(generation=x, multiprocessing_var=multiprocessing_var)
        else:
            pop.grade_single(generation=x)
        if x == 1:
            print("break")
        if pop.done:
            end = time.time()
            print("Finished at generation:", x, ", Population fistness:", pop.fitness_history[-1])
            print("Finished after:",end-start," Seconds")
            break
        else:
            pop.evolve()
            print("Finished with ",x,"Generation" )
        gc.collect()
        round_time.append(time.time())

#%%
    end_2 = time.time()
    # Plot fitness history
    if SHOW_PLOT:
        print("Showing fitness history graph")
        plt.plot(np.arange(len(pop.fitness_history)), pop.fitness_history)
        plt.ylabel('Fitness')
        plt.xlabel('Generations')
        plt.title('Fitness - pop_size {} mutate_prob {} retain {} random_retain {}'.format(pop_size, mutate_prob, retain, random_retain))
        plt.show()

        pop.save_gens_winner()
        scatterplot(pop.save_file)
        plot_histogram_all(pop.save_file)
    print("FINISCHED!!! after ", end_2-start, "seconds")
    print(round_time)
    tmp = start
    for i in range(0,len(round_time)):
        print("Round: ", i," Time: ", round_time[i] - tmp )
        tmp = round_time[i]
