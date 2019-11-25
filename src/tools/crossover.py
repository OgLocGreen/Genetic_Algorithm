
import random


######################################
# GA Crossovers                      #
######################################
# Some of these Functions are duplications of https://github.com/DEAP/deap/blob/master/deap/tools/crossover.py

def onepoint(father_gene, mother_gene):
    """Executes a one point crossover on the input :term:`sequence` individuals.
    The two individuals are modified in place. The resulting individuals will
    respectively have the length of the other.
    :param father_gene: The first individual participating in the crossover.
    :param mother_gene: The second individual participating in the crossover.
    :returns: A tuple of two individuals.
    This function uses the :func:`~random.randint` function from the
    python base :mod:`random` module.
    """
    size = min(len(father_gene), len(mother_gene))
    crossoverpoint = random.randint(1, size - 1)
    mother_gene[crossoverpoint:], father_gene[crossoverpoint:] = father_gene[crossoverpoint:], mother_gene[crossoverpoint:]
    return father_gene, mother_gene

def twopoint(father_gene, mother_gene):
    """Executes a two-point crossover on the input :term:`sequence`
    individuals. The two individuals are modified in place and both keep
    their original length.
    :param father_gene: The first individual participating in the crossover.
    :param mother_gene: The second individual participating in the crossover.
    :returns: A tuple of two individuals.
    This function uses the :func:`~random.randint` function from the Python
    base :mod:`random` module.
    """
    size = min(len(father_gene), len(mother_gene))
    crossoverpoint1 = random.randint(1, size)
    crossoverpoint2 = random.randint(1, size - 1)
    if crossoverpoint2 >= crossoverpoint1:
        crossoverpoint2 +=1
    elif crossoverpoint2 > crossoverpoint1:
        crossoverpoint1, crossoverpoint2 = crossoverpoint2, crossoverpoint1
    mother_gene[crossoverpoint1:crossoverpoint2], father_gene[crossoverpoint1:crossoverpoint2] = father_gene[crossoverpoint1:crossoverpoint2], mother_gene[crossoverpoint1:crossoverpoint2]
    return father_gene, mother_gene

def uniform(father_gene,mother_gene,mutation_prob):
    """Executes a uniform crossover that modify in place the two
    :term:`sequence` individuals. The attributes are swapped accordingto the
    *indpb* probability.
    :param father_gene: The first individual participating in the crossover.
    :param mother_gene: The second individual participating in the crossover.
    :param indpb: Independent probabily for each attribute to be exchanged.
    :returns: A tuple of two individuals.
    This function uses the :func:`~random.random` function from the python base
    :mod:`random` module.
    """
    size = min(len(father_gene), len(mother_gene))
    for i in range(size):
        if random.random() < mutation_prob:
            father_gene[i], mother_gene[i] = mother_gene[i], father_gene[i]

    return father_gene, mother_gene


def zufall(father_gene, mother_gene):
        return [random.choice(pixel_pair) for pixel_pair in zip(father_gene, mother_gene)]
