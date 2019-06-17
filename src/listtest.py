# set
pySet = {'e', 'a', 'u', 'o', 'i'}
print(reversed(sorted(pySet, reverse=True)))

# dictionary
pyDict = {'e': 1, 'a': 2, 'u': 3, 'o': 4, 'i': 5}
print(sorted(pyDict, reverse=True))

# frozen set
pyFSet = frozenset(('e', 'a', 'u', 'o', 'i'))
print(reversed(sorted(pyFSet, reverse=True)))

# set
pySet = {4,3,1,2,3}
print(sorted(pySet, reverse=False))
