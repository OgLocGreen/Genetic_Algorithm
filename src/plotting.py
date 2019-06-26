#%%
import numpy as np
import matplotlib.pyplot as plt
import json

def plot_winner():
    with open("./data.json", "r") as f:
        data = json.load(f)
    x = []
    y = []

    for gen in data["Winner"]:
        x.append(data["Winner"][gen]["acc"])
        y.append(data["Winner"][gen]["loss"])

    plt.scatter(x, y, s=80, marker="+")
    plt.xlabel('acc', fontsize=18)
    plt.ylabel('loss', fontsize=16)
    plt.gca().invert_yaxis()
    plt.show()

def plot_all():
    with open("./data.json", "r") as f:
        data = json.load(f)

    for pop in data:
        x = []
        y = []
        for gen in data[pop]:
            x.append(gen["acc"])    ## hier dürfte noch ein fehler sein
            y.append(gen["loss"])   ## hier dürfte noch ein feheler sein
        plt.scatter(x, y, s=80, marker="+")
        plt.xlabel('acc', fontsize=18)
        plt.ylabel('loss', fontsize=16)
        plt.gca().invert_yaxis()
        plt.show(num=gen)

