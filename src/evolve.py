### https://gist.githubusercontent.com/gabrielgarza/377a692eb819d4efdf9a13b03dcb2358/raw/3a0b50435b2269203c3a3992362e76c04f81ad13/evolve.py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import json
from KNN import train_and_evalu

from plotting import plot_winner, plot_all, plot_normalverteilung

import gc



class Individual(object):
    def __init__(self, learningrate, dropout, epoch, batchsize):
        self.gene = (learningrate, dropout, epoch, batchsize)
        self.var_acc = 0
        self.var_loss = 0

    def fitness(self):
        """
            Returns fitness of individual
            Fitness is the difference between
        """
        self.var_loss, self.var_acc = train_and_evalu(self.gene[0], self.gene[1], self.gene[2],self.gene[3])
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
            self.individuals.append(Individual(learningrate, dropout, epoch, batchsize))

    def grade(self, generation=None):
        """
            Grade the generation by getting the average fitness of its individuals
        """
        fitness_sum = 0
        i = 0
        for x in self.individuals:
            print("individuals: ", i)
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
            children.append(Individual(learningrate, dropout, epoch, batchsize))


        if len(self.parents) > 0:                   ##überprüfen ob eltern vorhanden
            while len(children) < target_children_size:     ##solange bis gewollte kinder auch da sind
                father = random.choice(self.parents)        ## vater random aus eltern auswählen
                mother = random.choice(self.parents)        ## mutte random aus eltern auswählen
                if father != mother:                           ## wenn vatter nicht gleich mutter
                    ## kinder gegen werden zufällig aus den genen von Mutter und Vatter ausgewählt
                    child_genes = [random.choice(pixel_pair) for pixel_pair in zip(father.gene, mother.gene)]
                    ## Mutation der Gene mit einem Faktor
                    for x in range(0, len(child_genes)):
                        tmp = child_genes[x]
                        mutation = random.uniform(0, self.mutate_prob)       ## Mutationsfaktor mutate prob
                        var = random.choice([True, False])      ## Funktion um Positive bzw. negative Mutation
                        if var == True:                         ## Mutation funktioniert in Prozent
                            tmp = tmp - (tmp * mutation)
                        else:
                            tmp = tmp + (tmp * mutation)
                        child_genes[x] = round(tmp,5)          ## anschließend runden auf 5 nachkommastellen
                    child = Individual(child_genes[0], child_genes[1], child_genes[2], child_genes[3])
                    children.append(child)
                else:
                    print("father == mother selection new parents")

            self.individuals = children       ##Kinder werden Individumen für nächste generation

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
            with open("data.json", "r") as f:
                data = json.load(f)
        except:
            data ={}
        i = 0
        self.individuals = list(sorted(self.individuals, key=lambda x: x.var_acc, reverse=True))  ##indiviuen noch mal nach fitness sotierten
        family_tree ={generations: {}}
        for x in pop.individuals:
            generation = {
            "name": i,
            "learningrate": x.gene[0],
            "dropout": x.gene[1],
            "epoch": x.gene[2],
            "batchsize": x.gene[3],
            "acc": x.var_acc,
            "loss": x.var_loss
            }
            family_tree[generations][i] = generation
            i += 1
        data.update(family_tree)
        with open("data.json", "w") as outfile:
            json.dump(data, outfile, indent=2)
        print("saved population gens into data.json")
    
    def save_gens_winner(self):
        with open("data.json", "r") as f:
            data = json.load(f)
        self.grade() #damit alle induviduals noch auf fittnes überprüft werden
        self.individuals = list(sorted(self.individuals, key=lambda x: x.var_acc, reverse=True))
        i = 0
        family_tree ={"Winner": {}}
        for x in pop.individuals:
            generation = {
                "name": i,
                "learningrate":x.gene[0],
                "dropout":x.gene[1],
                "epoch":x.gene[2],
                "batchsize":x.gene[3],
                "acc":x.var_acc,
                "loss":x.var_loss
            }
            family_tree["Winner"][i] = generation
            i+=1
        data.update(family_tree)
        with open("data.json","w") as outfile:
            json.dump(data,outfile,indent=2)
        print("saved winnerpopulation gens into data.json")

if __name__ == "__main__":
    pop_size = 20
    mutate_prob = 0.3
    retain = 0.5
    random_retain = 0.05

    pop = Population(pop_size=pop_size, mutate_prob=mutate_prob, retain=retain, random_retain=random_retain)

    SHOW_PLOT = True
    GENERATIONS = 5
    for x in range(GENERATIONS):
        pop.grade(generation=x)
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
    plot_winner()
    plot_normalverteilung()
    print("FINISCHED!!!")

