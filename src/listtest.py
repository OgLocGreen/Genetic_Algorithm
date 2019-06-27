import numpy as np
import random

chromoson = {"Winner": {}}

gen =  {"a":1, "b" : "ja", "c" : 3}
gen1 =  {"a": 2, "b" : "ja", "c" : 4}


for i in range(0,2):
    chromoson["Winner"][i] = gen
    chromoson["Winner"][i] = gen1

y=np.zeros(shape=(10,4))

for i in range(0,10):
    learningrate = random.uniform(0.0005, 0.1)
    dropout = random.uniform(0.05, 0.5)
    epoch = random.uniform(5, 10)
    batchsize = random.uniform(32, 64)
    y[i] = [learningrate,dropout,epoch,batchsize]

yy=y
"""
print("first")
for u in range(0,len(y)):
    print("orginal",y[i])
    for x in range(0,len(y[i])):
        tmp = y[i][x]
        mutation = random.uniform(0,0.3)
        print("mutation",mutation)
        tmp = tmp * mutation
        y[i][x] = tmp
    print("new",y[i])
"""

print("second")
for u in range(0,len(yy)):
    print("orignal",yy[u])
    for x in range(0,len(yy[u])):
        tmp = yy[u][x]
        mutation = random.uniform(0, 0.3)
        print("mutation",mutation)
        var = random.choice([True, False])
        if var == True:
            tmp = tmp - (tmp * mutation)
        else:
            tmp = tmp + (tmp * mutation)
        print("yy[i][x]",yy[u][x])
        print("tmp",tmp)
        yy[u][x] = round(tmp,4)
    print("mutation",yy[u])




"""
for i in range(0,len(y)):
    for j in range(0,len(y[i])):
        print(y[i][j])
"""