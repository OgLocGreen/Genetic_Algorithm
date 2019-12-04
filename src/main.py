import gridSearch
import geneticAlgorithm

print("Hallo")
gridSearch.main("mnist_digits", "small", 50, False, 2, True)
gridSearch.main("mnist_digits", "small", 50, False, 2, True)

geneticAlgorithm.main("mnist_digits", "small", 25, 2, False, 2, True)
geneticAlgorithm.main("mnist_digits", "small", 25, 2, False, 2, True)
print("finished")