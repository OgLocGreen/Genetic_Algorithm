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