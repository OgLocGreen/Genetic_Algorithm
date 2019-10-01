
import numpy as np
import matplotlib.pyplot as plt
import random
import multiprocessing
import json
import gc
import datetime
import os


import KNN
import crossover
import mutation
import individual
import selection

class Population(object):

    def __init__(self, pop_size=50, mutate_prob=0.01, retain=0.2, random_retain=0.03):
        """
            Args
                pop_size: size of population
                mutate_pron: standard deviation of random.gauss()
                retain: parents = polupation[:retain]
                random_retain: how many of the unfittest are retained
        """
        self.pop_size = pop_size
        self.mutate_prob = mutate_prob
        self.retain = retain
        self.random_retain = random_retain
        self.fitness_history = []
        self.parents = []
        self.done = False
        self.save_file = "{}.{}.{}.json".format(datetime.datetime.now().year,
                                                datetime.datetime.now().month,
                                                datetime.datetime.now().day)

        if os.path.isfile(self.save_file):
            for i in range(1, 10):
                self.save_file = "{}.{}.{}-{}.json".format(datetime.datetime.now().year,
                                                           datetime.datetime.now().month,
                                                           datetime.datetime.now().day,
                                                           i)
                if os.path.isfile(self.save_file) == False:
                    break

        # Create individuals
        self.individuals = []
        for x in range(pop_size):
            Neuronen_Layer1 = -1 
            Neuronen_Layer2 = -1 
            Neuronen_Layer3 = -1
            self.individuals.append(individual.Individual(Neuronen_Layer1, Neuronen_Layer2, Neuronen_Layer3))

    def grade_single(self, generation=None):

        # Grade the generation by getting the average fitness of its individuals

        fitness_sum = 0
        i = 0
        for x in self.individuals:
            fitness_sum += x.fitness()
            print("individual: ", i)
            print("fitness: ", x.var_acc)
            i = i + 1
        del i
        pop_fitness = fitness_sum / self.pop_size
        self.fitness_history.append(pop_fitness)
        self.save_gens(generation)
        # Set Done flag if we hit target
        if pop_fitness >= 0.90:
            self.done = True

        if generation is not None:
            if generation % 5 == 0:
                print("Episode", generation, "Population fitness:", pop_fitness)
                print("----------------------------------------")
                print("----------------------------------------")
        gc.collect()

    def grade_multi(self, generation=None, multiprocessing_var=2):
        """
            Grade the generation by getting the average fitness of its individuals with multiprocessing
        """
        fitness_sum = 0

        p = multiprocessing.Pool(processes=multiprocessing_var)

        accloss = p.map(fitness_multi, self.individuals)

        i = 0
        for x in self.individuals:
            x.var_acc, x.var_loss = accloss[i][0], accloss[i][1]
            fitness_sum += x.var_acc
            i += 1
        del i
        pop_fitness = fitness_sum / self.pop_size
        self.fitness_history.append(pop_fitness)
        self.save_gens(generation)
        # Set Done flag if we hit target
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
                father, mother = selection.selRoulette(self.parents,k=2)
                if father != mother:  ## wenn vatter nicht gleich mutter

                    child_gene_1, child_gene_2 = crossover.twopoint(father.gene, mother.gene)
                    
                    child_gene_1 = mutation.gauss(child_gene_1, self.mutate_prob)
                    child_gene_2 = mutation.gauss(child_gene_2, self.mutate_prob)

                    child_1 = individual.Individual(child_gene_1[0], child_gene_1[1], child_gene_1[2])
                    child_2 = individual.Individual(child_gene_2[0], child_gene_2[1], child_gene_2[2])
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
        i = 0
        self.individuals = list(sorted(self.individuals, key=lambda x: x.var_acc,
                                       reverse=True))  ##indiviuen noch mal nach fitness sotierten
        family_tree = {generations: {}}
        for x in self.individuals:
            generation = {
                "name": i,
                "Neuronen_Layer1": str(x.gene[0]),
                "Neuronen_Layer2": str(x.gene[1]),
                "Neuronen_Layer3": str(x.gene[2]),
                "acc": str(x.var_acc),
                "loss": str(x.var_loss)
            }
            family_tree[generations][i] = generation
            i += 1
        del i
        data.update(family_tree)
        del family_tree
        with open(self.save_file, "w") as outfile:
            json.dump(data, outfile, indent=2)
        print("saved population gens into {}".format(self.save_file))
        print(generation)
        gc.collect()

    def save_gens_winner(self):
        with open(self.save_file, "r") as f:
            data = json.load(f)
        self.grade_multi()  # damit alle induviduals noch auf fittnes überprüft werden
        self.individuals = list(sorted(self.individuals, key=lambda x: x.var_acc, reverse=True))
        i = 0
        family_tree = {"Winner": {}}
        for x in self.individuals:
            generation = {
                "name": i,
                "Neuronen_Layer1": str(x.gene[0]),
                "Neuronen_Layer2": str(x.gene[1]),
                "Neuronen_Layer3": str(x.gene[2]),
                "acc": str(x.var_acc),
                "loss": str(x.var_loss)
            }
            family_tree["Winner"][i] = generation
            i += 1
        del i
        data.update(family_tree)
        del family_tree
        with open(self.save_file, "w") as outfile:
            json.dump(data, outfile, indent=2)
        print("saved winnerpopulation gens into data.json")
        gc.collect()

def fitness_multi(individuum):
    """
        Returns fitness(accuarcy ) and loss of individual
    """
    var_loss, var_acc = KNN.train_and_evalu_model(individuum.gene[0], individuum.gene[1], individuum.gene[2])
    return var_acc, var_loss