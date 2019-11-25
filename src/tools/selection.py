import random
import operator

def selRandom(individuals, k):
    """Select *k* individuals at random from the input *individuals* with
    replacement. The list returned contains references to the input
    *individuals*.
    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :returns: A list of selected individuals.
    This function uses the :func:`~random.choice` function from the
    python base :mod:`random` module.
    """
    return [random.choice(individuals) for i in range(k)]


def selBest(individuals, k, fit_attr="fitness"):
    """Select the *k* best individuals among the input *individuals*. The
    list returned contains references to the input *individuals*.
    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :param fit_attr: The attribute of individuals to use as selection criterion
    :returns: A list containing the k best individuals.
    """
    return sorted(individuals, key=operator.attrgetter(fit_attr), reverse=True)[:k]

def selTournament(individuals, k, tournsize, fit_attr="var_acc"):
    """Select the best individual among *tournsize* randomly chosen
    individuals, *k* times. The list returned contains
    references to the input *individuals*.
    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :param tournsize: The number of individuals participating in each tournament.
    :param fit_attr: The attribute of individuals to use as selection criterion
    :returns: A list of selected individuals.
    This function uses the :func:`~random.choice` function from the python base
    :mod:`random` module.
    """
    chosen = []
    for i in range(k):
        aspirants = selRandom(individuals, tournsize)
        chosen.append(max(aspirants, key=operator.attrgetter(fit_attr)))
    return chosen

def selRoulette(individuals, k, fit_attr="var_acc"):
    """Select *k* individuals from the input *individuals* using *k*
    spins of a roulette. The selection is made by looking only at the first
    objective of each individual. The list returned contains references to
    the input *individuals*.
    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :param fit_attr: The attribute of individuals to use as selection criterion
    :returns: A list of selected individuals.
    This function uses the :func:`~random.random` function from the python base
    :mod:`random` module.
    .. warning::
       The roulette selection by definition cannot be used for minimization
       or when the fitness can be smaller or equal to 0.
    """

    s_inds = sorted(individuals, key=operator.attrgetter(fit_attr), reverse=True)
    sum_fits = sum(getattr(ind, fit_attr)for ind in individuals)
    chosen = []
    for i in range(k):
        u = random.random() * sum_fits
        sum_ = 0
        for ind in s_inds:
            sum_ += getattr(ind, fit_attr)
            if sum_ > u:
                chosen.append(ind)
                break

    return chosen[0],chosen[1]
