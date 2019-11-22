"""
coding=utf-8
"""
import gc
import time
import argparse

import numpy as np
import matplotlib.pyplot as plt

import plotting
import population




def main(dataset_arg="mnist_fashion", knn_size_arg="small", pop_size_arg=10,
         generations_arg=2, gpu_arg=False, multiprocessing_arg=2, small_dataset_args=False):
    """
        Args
            dataset_arg: which dataset to train on
            knn_size_arg: which size of neural Network
            pop_size_arg: how big of an population you wanna train
            generations_arg: how many generations to train
            gpu_arg: do u you wanna train on gpu
            multiprocessing_arg: how many processes only on cpu
    """
    pop_size = pop_size_arg
    mutate_prob = 0.1 #sigma for random.gauss()
    retain = 0.8
    random_retain = 0.05
    generations = generations_arg
    gpu = gpu_arg
    dataset = dataset_arg
    small_dataset = False
    knn_size = knn_size_arg

    multiprocessing_flag = True
    multiprocessing_var = multiprocessing_arg
    show_plot = False
    save_plot = True

    if multiprocessing_var < 1:
        multiprocessing_flag = False

    pop = population.Population(pop_size=pop_size, mutate_prob=mutate_prob, retain=retain,
                                random_retain=random_retain, generations=generations,
                                dataset=dataset, small_dataset=small_dataset,
                                knn_size=knn_size, gpu=gpu, multiprocessing=multiprocessing_var,
                                multiprocessing_flag=multiprocessing_flag)
    start = time.time()
    lastround = start
    round_time = []
    pop.save_init_data()
    for tmp_generation in range(generations):
        pop.grade(generation=tmp_generation)
        if pop.done:
            print("Finished at generation:", tmp_generation,
                  ", Population fistness:", pop.fitness_history[-1])
            break
        else:
            pop.evolve()
            print("Finished with ", tmp_generation, "Generation")
        newround = time.time()
        round_time.append(newround - lastround)
        lastround = newround
        gc.collect()

    end_2 = time.time()
    round_time.append(end_2 - lastround)
    pop.save_gens_winner()
    for i in range(0, len(round_time)):
        print("Round: ", i, " Time: ", round_time[i])
    pop.save_end_data(round_time)

#%%
    # Plot fitness history
    if show_plot:
        print("Showing fitness history graph")
        plt.plot(np.arange(len(pop.fitness_history)), pop.fitness_history)
        plt.ylabel('Fitness')
        plt.xlabel('Generations')
        plt.title('Fitness - pop_size {} mutate_prob {} retain {} random_retain {}'.format(
            pop_size, mutate_prob, retain, random_retain))
        plt.show()
    if save_plot:
        plotting.scatterplot(dir_path=pop.dir_path, save_file=pop.save_file, save = True)
        plotting.joint_plot(dir_path=pop.dir_path, save_file=pop.save_file, save = True)
    print("FINISCHED!!! after ", end_2-start, "seconds")
    print(round_time)




if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Genetic Algorithm")
    PARSER.add_argument('dataset', help='which dataset: mnist_fashion, mnist_digits, cifar10 ', type=str, default="mnist_fashion")
    PARSER.add_argument("knn_size", help='which size of KNN: small, big ', type=str, default="small")
    PARSER.add_argument("pop_size", help='Populationsize ', type=int, default=10)
    PARSER.add_argument("generations", help='how many Generations ', type=int, default=3)
    PARSER.add_argument("gpu", help='if gpu enable True or False ', type=bool, default=False)
    PARSER.add_argument("multiprocessing", help='How many parallel processes: 2, 4, if 0 then no multiprocess', type=int, default=2)
    PARSER.add_argument("small_dataset", help="to train with only 0.1 of the original Dataset", type=bool, default=False)
    ARGS = PARSER.parse_args()

    main(dataset_arg=ARGS.dataset, knn_size_arg=ARGS.knn_size,
         pop_size_arg=ARGS.pop_size, generations_arg=ARGS.generations,
         gpu_arg=ARGS.gpu, multiprocessing_arg=ARGS.multiprocessing, small_dataset_args=ARGS.small_dataset)
