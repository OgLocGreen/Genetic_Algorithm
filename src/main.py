import gridSearch
import geneticAlgorithm
import gc
if __name__ == '__main__':
    print("start")
    """
    gridSearch.main("mnist_digits", "small", 50, False, 2, True)
    gc.collect()
    gridSearch.main("mnist_digits", "small", 50, False, 2, False)
    gc.collect()
    geneticAlgorithm.main("mnist_digits", "small", 25, 2, False, 2, True)
    gc.collect()
    geneticAlgorithm.main("mnist_digits", "small", 25, 2, False, 2, False)
    gc.collect()

    gridSearch.main("mnist_digits", "big", 50, False, 2, True)
    gc.collect()
    gridSearch.main("mnist_digits", "big", 50, False, 2, False)
    gc.collect()
    geneticAlgorithm.main("mnist_digits", "big", 25, 2, False, 2, True)
    gc.collect()
    geneticAlgorithm.main("mnist_digits", "big", 25, 2, False, 2, False)
    gc.collect()

    gridSearch.main("mnist_digits", "small", 250, False, 2, True)
    gc.collect()
    gridSearch.main("mnist_digits", "small", 250, False, 2, False)
    gc.collect()
    geneticAlgorithm.main("mnist_digits", "small", 50, 5, False, 2, True)
    gc.collect()
    geneticAlgorithm.main("mnist_digits", "small", 50, 5, False, 2, False)
    gc.collect()
    gridSearch.main("mnist_digits", "big", 250, False, 2, True)
    gc.collect()
    gridSearch.main("mnist_digits", "big", 250, False, 2, False)
    gc.collect()

    geneticAlgorithm.main("mnist_digits", "big", 50, 5, False, 2, True)
    gc.collect()
    geneticAlgorithm.main("mnist_digits", "big", 50, 5, False, 2, False)
    gc.collect()

    
    gridSearch.main("cifar10", "big", 50, False, 2, True)
    gc.collect()
    """
    gridSearch.main("cifar10", "big", 50, False, 2, False)
    gc.collect()
    geneticAlgorithm.main("cifar10", "big", 25, 2, False, 2, True)
    gc.collect()
    geneticAlgorithm.main("cifar10", "big", 25, 2, False, 2, False)
    gc.collect()

    gridSearch.main("cifar10", "big", 250, False, 2, True)
    gc.collect()
    gridSearch.main("cifar10", "big", 250, False, 2, False)
    gc.collect()
    geneticAlgorithm.main("cifar10", "big", 50, 5, False, 2, True)
    gc.collect()
    geneticAlgorithm.main("cifar10", "big", 50, 5, False, 2, False)
    gc.collect()


    print("finished")