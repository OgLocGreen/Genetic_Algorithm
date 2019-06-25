import numpy as np

chromoson = {"Winner": {}}

gen =  {"a":1, "b" : "ja", "c" : 3}
gen1 =  {"a": 2, "b" : "ja", "c" : 4}


for i in range(0,2):
    chromoson["Winner"][i] = gen
    chromoson["Winner"][i] = gen1

print(chromoson)
#y = [[1,2,3,4,5],[5,4,3,2,1]]
y = []
y[0] = [1,2,3,4,5]
y[1] = [5,4,3,2,1]


for i in range(0,len(y)):
    for j in range(0,len(y[i])):
        print(y[i][j])