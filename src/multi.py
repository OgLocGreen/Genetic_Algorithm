import time
import threading
import multiprocessing
import random


def calc_square(numbers):
    print("calculate square numbers")
    for n in numbers:
        time.sleep(1)
        print('square:',n*n)

def calc_cube(numbers):
    print("calculate cube of numbers")
    for n in numbers:
        time.sleep(1)
        print('cube:',n*n*n)

def thread():

    arr = [2,3,8,9]

    t = time.time()

    t1= threading.Thread(target=calc_square, args=(arr,))
    t2= threading.Thread(target=calc_cube, args=(arr,))

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    print("done in : ",time.time()-t)
    print("Hah... I am done with all my work now!")

def process():
    arr = [2, 3, 8]
    p1 = multiprocessing.Process(target=calc_square, args=(arr,))
    p2 = multiprocessing.Process(target=calc_cube, args=(arr,))

    p1.start()
    p2.start()

    p1.join()
    p2.join()

    print("Done!")

class Individual(object):
    def __init__(self, learningrate, dropout, epoch, batchsize):
        self.gene = (learningrate, dropout, epoch, batchsize)
        self.var_acc = 0
        self.var_loss = 0


def fitness(Individual):
    learningrate = Individual.gene[0]
    var_acc = learningrate
    return var_acc


def multipool():

    population = []
    for n in range(3):
        learningrate = n
        dropout = random.uniform(0.05, 0.5)
        epoch = random.uniform(5, 10)
        batchsize = random.uniform(32, 64)
        population.append(Individual(learningrate, dropout, epoch, batchsize))

    i = 0
    for x in population:
        #fitness(x)
        i  += 1
    for x in population:
        print(x.var_acc)
    print("--- Multiprocess---")
    p = multiprocessing.Pool()
    print(population)

    result = p.map(fitness, population)

    population[0].var_acc,population[1].var_acc,population[2].var_acc = result
    for x in population:
        print(x.var_acc)





if __name__ == "__main__":
    multipool()