### https://gist.githubusercontent.com/gabrielgarza/377a692eb819d4efdf9a13b03dcb2358/raw/3a0b50435b2269203c3a3992362e76c04f81ad13/evolve.py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import multiprocessing
import json

from KNN import train_and_evalu

from plotting import plot_winner, plot_all, plot_histogram_all, scatterplot

import gc
import datetime


#%%
class Individual(object):
    def __init__(self, learningrate, dropout, epoch, batchsize, optimizer):
        self.gene = (learningrate, dropout, epoch, batchsize, optimizer)
        self.var_acc = 0
        self.var_loss = 0

    def fitness(self):
        """
            Returns fitness of individual
            Fitness is the difference between
        """
        self.var_loss, self.var_acc = train_and_evalu(self.gene[0], self.gene[1], self.gene[2], self.gene[3], self.gene[4])
        return self.var_acc



class Population(object):

    def __init__(self, pop_size=50, mutate_prob=0.01, retain=0.2, random_retain=0.03):
        """
            Args
                pop_size: size of population
                fitness_goal: goal that population will be graded against
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

        # Create individuals
        self.individuals = []
        for x in range(pop_size):
            """
            learningrate = random.uniform(learningrate_min, learningrate_max)
            dropout = random.uniform(dropout_min, dropout_max)
            epoch = random.uniform(epoch_min epoch_max)
            batchsize = random.uniform(batchsize_min batchsize_max)
            """
            learningrate = random.uniform(0.0005, 0.1)
            dropout = random.uniform(0.05, 0.5)
            epoch = random.uniform(5, 10)
            batchsize = random.uniform(32, 64)
            optimizer = random.uniform(0, 3)
            self.individuals.append(Individual(learningrate, dropout, epoch, batchsize, optimizer))

    def grade_single(self, generation=None):

        #Grade the generation by getting the average fitness of its individuals

        fitness_sum = 0
        i = 0
        for x in self.individuals:
            fitness_sum += x.fitness()
            print("individual: ", i)
            print("fitness: ",x.var_acc)
            i = i+1
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

    def grade_multi(self, generation=None):
        """
            Grade the generation by getting the average fitness of its individuals
        """
        fitness_sum = 0

        p = multiprocessing.Pool(processes=2)

        #print(p.map(fitness_multi,self.individuals))
        accloss = p.map(fitness_multi,self.individuals)

        i=0
        for x in self.individuals:
            x.var_acc, x.var_loss = accloss[i][0], accloss[i][1]
            fitness_sum += x.var_acc
            i += 1
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

    def crossover_random(self, father_gene, mother_gene):
        return [random.choice(pixel_pair) for pixel_pair in zip(father_gene, mother_gene)]
    
    def mutation_random(self, child_genes):
        for x in range(0, len(child_genes)):
            tmp = child_genes[x]
            mutation = random.uniform(0, self.mutate_prob)       ## Mutationsfaktor mutate prob
            var = random.choice([True, False])      ## Funktion um Positive bzw. negative Mutation
            if var is True:                         ## Mutation funktioniert in Prozent
                tmp = tmp - (tmp * mutation)
            else:
                tmp = tmp + (tmp * mutation)
            child_genes[x] = round(tmp,5)
        print(child_genes)
        return child_genes

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

    def breed(self):
        """
            Crossover the parents to generate children and new generation of individuals
        """
        target_children_size = self.pop_size - 4     ##Auswählen wie viele kinder erstellt 2 werden mit den besten 2 eltern ersetzt
                                                     ## zwei werden random hinzu gefügtvlt des ganze nicht hart mit 2 sondern mit variablen
        children = []
        children.append(self.parents[0])      ##zwei besten in nächste pop hinzu fügen
        children.append((self.parents[1]))
        for i in range(0,2):                        ##zwei random in nächste pop hinzu fügen
            learningrate = random.uniform(0.0005, 0.1)
            dropout = random.uniform(0.05, 0.5)
            epoch = random.uniform(5, 10)
            batchsize = random.uniform(32, 64)
            optimizer = random.uniform(0, 3)
            children.append(Individual(learningrate, dropout, epoch, batchsize, optimizer))


        if len(self.parents) > 0:                   ##überprüfen ob eltern vorhanden
            while len(children) < target_children_size:     ##solange bis gewollte kinder auch da sind
                father = random.choice(self.parents)        ## vater random aus eltern auswählen
                mother = random.choice(self.parents)        ## mutte random aus eltern auswählen
                if father != mother:                           ## wenn vatter nicht gleich mutter
                    ## kinder gegen werden zufällig aus den genen von Mutter und Vatter ausgewählt
                    child_gene = self.crossover_random(father.gene, mother.gene)
                    ## Mutation der Gene mit einem Faktor
                    child_gene_new = self.mutation_random(child_gene)

                    child = Individual(child_gene_new[0], child_gene_new[1], child_gene_new[2], child_gene_new[3], child_gene_new[4])
                    children.append(child)
                else:
                    print("father == mother selection new parents")

            self.individuals = children       # Kinder werden Individumen für nächste generation
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

    def save_gens(self,generations):
        try :
            with open(self.save_file, "r") as f:
                data = json.load(f)
        except:
            data = {}
        i = 0
        self.individuals = list(sorted(self.individuals, key=lambda x: x.var_acc, reverse=True))  ##indiviuen noch mal nach fitness sotierten
        family_tree ={generations: {}}
        for x in pop.individuals:
            generation = {
                "name": i,
                "learningrate": str(x.gene[0]),
                "dropout": str(x.gene[1]),
                "epoch": str(x.gene[2]),
                "batchsize": str(x.gene[3]),
                "optimizer": str(x.gene[4]),
                "acc": str(x.var_acc),
                "loss": str(x.var_loss)
            }
            family_tree[generations][i] = generation
            i += 1
        data.update(family_tree)
        with open(self.save_file, "w") as outfile:
            json.dump(data, outfile, indent=2)
        print("saved population gens into {}".format(self.save_file))
    
    def save_gens_winner(self):
        with open(self.save_file, "r") as f:
            data = json.load(f)
        self.grade_single() #damit alle induviduals noch auf fittnes überprüft werden
        self.individuals = list(sorted(self.individuals, key=lambda x: x.var_acc, reverse=True))
        i = 0
        family_tree ={"Winner": {}}
        for x in pop.individuals:
            generation = {
                "name": i,
                "learningrate": str(x.gene[0]),
                "dropout": str(x.gene[1]),
                "epoch": str(x.gene[2]),
                "batchsize": str(x.gene[3]),
                "optimizer": str(x.gene[4]),
                "acc": str(x.var_acc),
                "loss": str(x.var_loss)
            }
            family_tree["Winner"][i] = generation
            i+=1
        data.update(family_tree)
        with open(self.save_file, "w") as outfile:
            json.dump(data, outfile, indent=2)
        print("saved winnerpopulation gens into data.json")


def fitness_multi(individuum):
    """
        Returns fitness of individual
        Fitness is the difference between
    """
    var_loss, var_acc = train_and_evalu(individuum.gene[0], individuum.gene[1], individuum.gene[2],individuum.gene[3],individuum.gene[4])
    return var_acc, var_loss

if __name__ == "__main__":
    pop_size = 10
    mutate_prob = 0.3
    retain = 0.5
    random_retain = 0.05

    SHOW_PLOT = True
    GENERATIONS = 5
    multiprocessing_flag = False


    pop = Population(pop_size=pop_size, mutate_prob=mutate_prob, retain=retain, random_retain=random_retain)

    for x in range(GENERATIONS):
        if multiprocessing_flag:
            pop.grade_multi(generation=x)
        else:
            pop.grade_single(generation=x)
        if pop.done:
            print("Finished at generation:", x, ", Population fistness:", pop.fitness_history[-1])
            break
        else:
            pop.evolve()
        gc.collect()

#%%
    SHOW_PLOT = True
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
    print("FINISCHED!!!")

