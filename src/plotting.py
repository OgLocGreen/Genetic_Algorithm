import numpy as np
import matplotlib.pyplot as plt
import json

#%%
# Fixing random state for reproducibility
with open("./data.json", "r") as f:
    data = json.load(f)
x = []
y = []

for gen in data["Winner"]:
    x.append(gen["acc"])
    y.append(gen["loss"])
#%%
plt.scatter(x, y, s=80, marker="+")
plt.show()


#%%
xx =[]
for pop in data:
    for gen in pop:
        xx