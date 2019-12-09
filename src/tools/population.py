# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import random
import multiprocessing
import json
import gc
import datetime
import os
import sys 
import socket

from tools import KNN
from tools import crossover
from tools import mutation
from tools import individual
from tools import selection

from tools import evaluation

class Population(object):

    def __init__(self, pop_size=50, mutate_prob=0.01, retain=0.2, random_retain=0.03,
                 generations=5, dataset="mnist_fashion", knn_size="small", small_dataset=False,
                 gpu=False, multiprocessing=2, multiprocessing_flag=True):
        """
            Args
                pop_size: size of population
                mutate_pron: standard deviation of random.gauss()
                retain: parents = polupation[:retain]
                random_retain: how many of the unfittest are retained
                generations: how many generations to train
                dataset: which dataset to train on
                knn_size: size of the nn 
                small_dataset: should the dataset get downsized
                gpu: if u wanna train on the GPU
                multiprocessing: how many processes -> GPU is only one possible#
                multiprocessing_flag: do u wanna use multiprocessing 
        """
        self.gpu = gpu
        self.pop_size = pop_size
        self.knn_size = knn_size
        self.generations = generations
        self.mutate_prob = mutate_prob
        self.retain = retain
        self.random_retain = random_retain
        self.fitness_history = []
        self.parents = []
        self.done = False
        self.multiprocessing_flag = multiprocessing_flag
        self.small_dataset = small_dataset
        self.dataset = dataset
        self.multiprocessing = multiprocessing
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self.save_file = os.path.join(self.dir_path,
                                      "../../data/{}.{}.{}.{}.{}.{}.{}.{}.json".format(datetime.datetime.now().year,
                                                                        datetime.datetime.now().month,
                                                                        datetime.datetime.now().day,
                                                                        "GA", self.pop_size*self.generations,
                                                                        self.dataset, self.small_dataset, self.knn_size))

        if os.path.isfile(self.save_file):
            for i in range(1, 10):
                self.save_file = os.path.join(self.dir_path,
                                              "../../data/{}.{}.{}.{}.{}.{}.{}.{}-{}.json".format(datetime.datetime.now().year,
                                                                        datetime.datetime.now().month,
                                                                        datetime.datetime.now().day,
                                                                        "GA", self.pop_size*self.generations,
                                                                        self.dataset, self.small_dataset, self.knn_size, i))
                if os.path.isfile(self.save_file) == False:
                    break
        #self.save_file = filename = os.path.abspath(os.path.realpath(self.save_file))
        # Create individuals
        self.individuals = []
        for x in range(pop_size):
            self.individuals.append(individual.Individual(learningrate=-1, dropout=-1, epoch=-1, batchsize=-1, optimizer=-1))

    def grade(self, generation=None):
        """
            Grade the generation by getting the average fitness of its individuals with multiprocessing
        """
        i = 0
        fitness_sum = 0
        if self.multiprocessing_flag:
            p = multiprocessing.Pool(processes=self.multiprocessing)
            accloss = p.map(self.fitness, self.individuals)
            for x in self.individuals:
                x.var_loss, x.var_acc, x.variables = accloss[i][0], accloss[i][1], accloss[i][2]
                fitness_sum += x.var_acc
                i += 1
            del i
        else:
            for x in self.individuals:
                x.var_loss, x.var_acc, x.variables = self.fitness(x)
                fitness_sum += x.var_acc   
        pop_fitness = fitness_sum / self.pop_size
        self.fitness_history.append(pop_fitness)
        self.save_gens(generation)
        # Set Done flag if we hit target2
        if pop_fitness >= 0.90:
            self.done = True

        if generation is not None:
            if generation % 5 == 0:
                print("Episode", generation, "Population fitness:", pop_fitness)
                print("----------------------------------------")
                print("----------------------------------------")
        gc.collect()

    def select_parents(self):
        """
            Select the fittest individuals to be the parents of next generation (lower fitness it better in this case)
            Also select a some random non-fittest individuals to help get us out of local maximums
        """
        # Sort individuals by fitness (we use reversed because in this case lower fintess is better)
        self.individuals = list(sorted(self.individuals, key=lambda x: x.var_acc, reverse=True))
        # Keep the fittest as parents for next gen
        retain_length = self.retain * len(self.individuals)
        self.parents = self.individuals[:int(retain_length)]

        # Randomly select some from unfittest and add to parents array
        unfittest = self.individuals[int(retain_length):]
        for unfit in unfittest:
            if self.random_retain > np.random.rand():
                self.parents.append(unfit)
        gc.collect()

    def breed(self):
        """
            Crossover the parents to generate children and new generation of individuals
            nicht nur crossover sondern auch Eltern Selektion und Mutation
            Zusätzlich gibt es noch besonderheiten wie übernehmmen der besten 2 individuen in die nächste pop

        """
        target_children_size = self.pop_size - 2  ##Auswählen wie viele kinder erstellt 2 werden mit den besten 2 eltern ersetzt
        children = []
        children.append(self.parents[0])  ##zwei besten in nächste pop hinzu fügen
        children.append((self.parents[1]))

        if len(self.parents) > 0:  ##überprüfen ob eltern vorhanden
            while len(children) < target_children_size:  ##solange bis gewollte kinder auch da sind
                father, mother = selection.selRoulette(self.parents, k=2)
                if father != mother:  ## wenn vatter nicht gleich mutter

                    child_gene_1, child_gene_2 = crossover.twopoint(father.gene, mother.gene)
                    
                    child_gene_1 = mutation.gauss(child_gene_1, self.mutate_prob)
                    child_gene_2 = mutation.gauss(child_gene_2, self.mutate_prob)

                    child_1 = individual.Individual(child_gene_1[0], child_gene_1[1], child_gene_1[2],
                                                    child_gene_1[3], child_gene_1[4])
                    child_2 = individual.Individual(child_gene_2[0], child_gene_2[1], child_gene_2[2],
                                                    child_gene_2[3], child_gene_2[4])
                    children.append(child_1)
                    children.append(child_2)
                else:
                    print("father == mother selection new parents")
        self.individuals = children  # Kinder werden Individumen für nächste generation
        del children
        gc.collect()

    def evolve(self):
        # 1. Select fittest
        self.select_parents()
        # 2. Create children and new generation
        self.breed()
        # 3. Reset parents and children
        self.parents = []
        self.children = []

    def save_gens(self, generations):
        try:
            with open(self.save_file, "r") as f:
                data = json.load(f)
        except:
            data = {}
        self.individuals = list(sorted(self.individuals, key=lambda x: x.var_acc,
                                       reverse=True))  ##indiviuen noch mal nach fitness sotierte
        family_tree = {generations: {}}
        i = 0
        for i, x in enumerate(self.individuals):
            generation = {
                "name": i,
                "learningrate": str(x.gene[0]),
                "dropout": str(x.gene[1]),
                "epoch": str(x.gene[2]),
                "batchsize": str(x.gene[3]),
                "optimizer": str(x.gene[4]),
                "acc": str(x.var_acc),
                "loss": str(x.var_loss),
                "variables" : str(x.variables)
            }
            family_tree[generations][i] = generation
        data["generation"].update(family_tree)
        with open(self.save_file, "w") as outfile:
            json.dump(data, outfile, indent=2)
        print("saved population gens into {}".format(self.save_file))
        print(generation)
        del data
        del family_tree
        gc.collect()

    def save_gens_winner(self):
        with open(self.save_file, "r") as f:
            data = json.load(f)
        self.grade()  # damit alle induviduals noch auf fittnes überprüft werden
        self.individuals = list(sorted(self.individuals, key=lambda x: x.var_acc, reverse=True))
        family_tree = {"Winner": {}}
        for i, x in enumerate(self.individuals):
            if i == 0:
                test_loss, test_acc, variables, precision_score_var, recall_score_var, f1_score_var, cm = KNN.train_and_evalu(gene=x.gene, dataset=self.dataset,
                                                                        knn_size=self.knn_size, small_dataset=self.small_dataset, gpu = self.gpu, f1=True)
                exel_path = os.path.join(self.dir_path, "../../data/evaluation.xlsx")
                print(exel_path)                                                 
                evaluation.write_cell(path_to_file=os.path.abspath(os.path.realpath(exel_path)), small_dataset=self.small_dataset, 
                dataset=self.dataset, knn_size=self.knn_size, iterations=(self.generations * self.pop_size), 
                algorithmus="GA", acc=test_acc, precision_score_var=precision_score_var, recall_score_var=recall_score_var, f1_score_var=f1_score_var)
            generation = {
                "name": i,
                "learningrate": str(x.gene[0]),
                "dropout": str(x.gene[1]),
                "epoch": str(x.gene[2]),
                "batchsize": str(x.gene[3]),
                "optimizer": str(x.gene[4]),
                "acc": str(x.var_acc),
                "loss": str(x.var_loss),
                "variables" : str(x.variables)
            }
            
            family_tree["Winner"][i] = generation
        data["generation"].update(family_tree)
        with open(self.save_file, "w") as outfile:
            json.dump(data, outfile, indent=2)
        print("saved population gens into {}".format(self.save_file))
        print(generation)
        del data
        del family_tree
        gc.collect()

    def save_init_data(self):
        try:
            with open(self.save_file, "r") as f:
                data = json.load(f)
        except:
            data = {}

        configurations = { "config": {
            
            "dataset" : self.dataset,
            "knn_size" : self.knn_size,
            "small_dataset" : self.small_dataset,
            "pop_size": self.pop_size,
            "Muationrate": self.mutate_prob,
            "Retain":self.retain,
            "random_retain" : self.random_retain,
            "generations" : self.generations,
            "PC_name":  socket.gethostname(),
            "Multiprocessing" : self.multiprocessing,
            "gpu" : self.gpu}
        }
        generation = {"generation":{}}
        round_time = {"round_time":{}}
        round_time = {"fitness_history":{}}
        data.update(configurations)
        data.update(generation)
        data.update(round_time)
        with open(self.save_file, "w") as outfile:
            json.dump(data, outfile, indent=2)
        del configurations
        del generation
    
    def save_end_data(self, round_time):
        try:
            with open(self.save_file, "r") as f:
                data = json.load(f)
        except:
            data = {}
        data["round_time"] = round_time
        data["fitness_history"] = self.fitness_history
        with open(self.save_file, "w") as outfile:
            json.dump(data, outfile, indent=2)

    def fitness(self,individuum):
        var_loss, var_acc, variables = KNN.train_and_evalu(gene=individuum.gene, dataset=self.dataset,
         knn_size=self.knn_size, small_dataset=self.small_dataset, gpu = self.gpu)
        return var_loss, var_acc, variables

