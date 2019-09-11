
from KNN import train_and_evalu
from plotting import plot_winner, plot_all, plot_histogram_all, scatterplot
import KNN
import crossover
import mutation



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