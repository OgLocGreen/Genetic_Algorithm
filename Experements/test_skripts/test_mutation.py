import random
import numpy as np

def gauss(child_genes,mutate_prob):
    for x in range(0,len(child_genes)):
        mutation = random.gauss(0, mutate_prob)  ## Mutationsfaktor mutate prob
        new = mutation * child_genes[x]
    return round(child_genes[x] + new , 5)

def gauss_2(child_genes,mutate_prob):
    for x in range(0,len(child_genes)):
        mutation = random.gauss(child_genes[x], mutate_prob)  ## Mutationsfaktor mutate prob    
        
        child_genes[x] = round(mutation, 5)
    return child_genes

def  numy_norm(child_genes,mutate_prob):
    for x in range(0, len(child_genes)):
        mutation = np.random.normal(loc = child_genes[x], scale= mutate_prob)

        child_genes[x] = round(mutation,5)
    return child_genes


a = [0.001,10]
gauss(a,0.1)
print(a)

a = [0.001,10]
gauss_2(a,0.1)
print(a)

a = [0.001,10]
numy_norm(a, 0.1)
print(a)



a = [0,1,2,3]
numy_norm(a, 0.1)
print(a)
print(np.round(a))


a = [0,1,2,3]
gauss_2(a, 0.1)
print(a)
print(np.round(a))


a = [0.5,32,50,100,64]
numy_norm(a, 1)
print(a)
print(np.round(a))


a = [0.5,32,50,100,64]
gauss_2(a, 1)
print(a)
print(np.round(a))