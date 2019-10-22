import KNN
import random

class Individual(object):
    def __init__(self, learningrate, dropout, epoch, batchsize, optimizer):
        if not (0.0005 < learningrate < 0.1):
            learningrate = random.uniform(0.0005, 0.1)
        if not (0.05 < dropout < 0.5):
            dropout = random.uniform(0.05, 0.5)
        if not (50 < epoch < 100):
            epoch = random.randint(50, 100)
        if not (16 < batchsize < 64): 
            batchsize = random.randint(32, 64)
        if not (-0.5 < optimizer < 3.5):
            optimizer = random.uniform(-0.5, 3.5)
        self.gene = [learningrate, dropout, epoch, batchsize, optimizer]
        print("gene: ", self.gene)
        self.var_acc = 0
        self.var_loss = 0
        self.variables = 0

    

    def fitness(self):
        """
            Returns fitness of individual
            Fitness is the difference between
        """
        self.var_loss, self.var_acc = KNN.train_and_evalu_cifar10_mean(self.gene[0], self.gene[1], self.gene[2], self.gene[3], self.gene[4])
        return self.var_loss, self.var_acc