import random


def zufall(child_genes,mutate_prob):
    for x in range(0, len(child_genes)):
        tmp = child_genes[x]
        mutation = random.uniform(0, mutate_prob)  ## Mutationsfaktor mutate prob
        var = random.choice([True, False])  ## Funktion um Positive bzw. negative Mutation
        if var is True:  ## Mutation funktioniert in Prozent
            tmp = tmp - (tmp * mutation)
        else:
            tmp = tmp + (tmp * mutation)
        child_genes[x] = round(tmp, 5)
    print(child_genes)
    return child_genes

def gauss(child_genes,mutate_prob):
    for x in range(0,len(child_genes)):
        mutation = random.gauss(0, mutate_prob)  ## Mutationsfaktor mutate prob
        new = mutation * child_genes[x]
        child_genes[x] = round(child_genes[x] + new , 5)
    return child_genes


def gauss_2(child_genes,mutate_prob, indpb):
    for x in range(0,len(child_genes)):
        if random.random() < indpb:
            mutation = random.gauss(0, mutate_prob)  ## Mutationsfaktor mutate prob
            new = mutation * child_genes[x]
            child_genes[x] = round(child_genes[x] + new , 5)
    return child_genes
