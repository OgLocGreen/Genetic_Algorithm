import random

def gauss(child_genes,mutate_prob):
    for x in range(0,len(child_genes)):
        mutation = random.gauss(0, mutate_prob)  ## Mutationsfaktor mutate prob
        new = mutation * child_genes[x]
    return round(child_genes[x] + new , 5)

def gauss_3(child_genes,mutate_prob):
    for x in range(0,len(child_genes)):
        mutation = random.gauss(child_genes[x], mutate_prob)  ## Mutationsfaktor mutate prob    
        
        child_genes[x] = round(mutation, 5)
    return child_genes

a = [0.001,10]
mutation_pb = 1
gauss(a,0.1)
print(a)

a = [0.001,10]
mutation_pb = 1
gauss_3(a,0.1)
print(a)