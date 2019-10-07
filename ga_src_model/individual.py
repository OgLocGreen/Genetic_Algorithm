import KNN
import random

class Individual(object):
    def __init__(self, Neuronen_Layer1,Neuronen_Layer2,Neuronen_Layer3):
        if not (1 < Neuronen_Layer1 < 257):
            Neuronen_Layer1 = random.randint(1,257)  
        if not (1 < Neuronen_Layer2 < 257):
            Neuronen_Layer2 = random.randint(1,257)  
        if not (1  < Neuronen_Layer3 < 257):
            Neuronen_Layer3 = random.randint(1,257)  
        self.gene = [Neuronen_Layer1, Neuronen_Layer2, Neuronen_Layer3]
        print("gene: ", self.gene)
        self.var_acc = 0
        self.var_loss = 0
        self.time_predict = 0
        self.fitness = 0
        

    def fitness_function(self):
        """
            Returns fitness of individual
            Fitness is the difference between
        """
        self.var_loss, self.var_acc = KNN.train_and_evalu_model(self.gene[0], self.gene[1], self.gene[2])
        return self.var_acc