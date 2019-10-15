
import json
import matplotlib.mlab as mlab
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

import datetime

sns.set(color_codes=True)




def plot_winner(file):
    with open(file, "r") as f:
        data = json.load(f)
    x = []
    y = []
    xmin = 0.8
    xmax = 0.9
    ymin = 0.8
    ymax = 0.2

    for gen in data["Winner"]:
        x.append(data["Winner"][gen]["acc"])
        y.append(data["Winner"][gen]["loss"])

    plt.scatter(x, y, s=80, marker="+")
    plt.xlabel('acc', fontsize=18)
    plt.ylabel('loss', fontsize=16)
    plt.gca().invert_yaxis(),
    axes = plt.gca()
    axes.set_xlim([xmin, xmax])
    axes.set_ylim([ymin, ymax])
    plt.show()

def plot_all(file):
    with open(file, "r") as f:
        data = json.load(f)

    for pop in data:
        x = []
        y = []
        for individum in data[pop]:
            try:
                x.append(float(data[pop][individum]["acc"]))    ## hier dürfte noch ein fehler sein
                y.append(float(data[pop][individum]["loss"]))   ## hier dürfte noch ein feheler sein
            except:
                print(pop,individum)
        plt.scatter(x, y, s=80, marker="+")
        plt.xlabel('acc', fontsize=18)
        plt.ylabel('loss', fontsize=16)
        plt.gca().invert_yaxis()
        plt.show()


def plot_histogram(title,werteliste):
    sns.distplot(werteliste, bins=10, kde=True, fit=stats.norm, rug=True)
    plt.title(title)

    # Get the fitted parameters used by sns
    (mu, sigma) = stats.norm.fit(werteliste)

    # Legend and labels
    plt.ylabel('Frequency')
    plt.legend(["normal dist. fit ($\mu=${0:.2g}, $\sigma=${1:.2f})".format(mu, sigma),
                "Gaußsche Kerneldichteabchätzung"])
    plt.show()


def plot_small_histogram(title, werteliste):
    sns.distplot(werteliste, bins=10, kde= False)
    plt.title(title)

    # Get the fitted parameters used by sns
    (mu, sigma) = stats.norm.fit(werteliste)

    # Legend and labels
    plt.ylabel('Frequency')
    plt.legend(["normal dist. fit ($\mu=${0:.2g}, $\sigma=${1:.2f})".format(mu, sigma),
                "Gaußsche Kerneldichteabchätzung"])
    plt.show()


def plot_histogram_all(file):
    with open(file, "r") as f:
        data = json.load(f)

    learningrate=[]
    batchsize=[]
    dropout=[]
    epoch =[]
    optimizer = []

    anzahl = 0
    for individum in data["Winner"]:
        learningrate.append(float(data["Winner"][individum]["learningrate"]))
        batchsize.append(float(data["Winner"][individum]["batchsize"]))
        dropout.append(float(data["Winner"][individum]["dropout"]))
        epoch.append(float(data["Winner"][individum]["epoch"]))
        optimizer.append(float(data["Winner"][individum]["optimizer"]))
        anzahl += 1
    
    print('learningrate mean=%.5f median=%.5f stdv=%.5f' % (np.mean(learningrate),np.median(learningrate), np.std(learningrate)))
    plot_histogram("learningrate",learningrate)
    plot_small_histogram("learningrate",learningrate)
    print('batchsize mean=%.5f median=%.5f stdv=%.5f' % (np.mean(batchsize), np.median(batchsize), np.std(batchsize)))
    plot_histogram("batchsize",batchsize)
    plot_small_histogram("batchsize",batchsize)
    print('dropout mean=%.5f median=%.5f stdv=%.5f' % (np.mean(dropout), np.median(dropout), np.std(dropout)))
    plot_histogram("dropout",dropout)
    print('epoch mean=%.5f median=%.5f stdv=%.5f' % (np.mean(epoch), np.median(epoch), np.std(epoch)))
    plot_histogram("epoch",epoch)
    print('optimizer mean=%.5f median=%.5f stdv=%.5f' % (np.mean(optimizer), np.median(optimizer), np.std(optimizer)))
    plot_histogram("optimizer", optimizer)


def scatterplot(file,yscale_log=False):
    x_label = "acc"
    y_label = "loss"

    # Create the plot object
    _, ax = plt.subplots()

    with open(file, "r") as f:
        data = json.load(f)

    for pop in data:
        x = []
        y = []
        if pop in ("1", "2", "3","4", "Winner"):
            for individum in data[pop]:
                try:
                    x.append(float(data[pop][individum]["acc"]))  ## hier dürfte noch ein fehler sein
                    y.append(float(data[pop][individum]["loss"]))  ## hier dürfte noch ein feheler sein
                except:
                    print("error")

            # Plot the data, set the size (s), color and transparency (alpha)
            # of the points
            ax.scatter(x, y, s=60, alpha=0.7)

            if yscale_log == True:
                ax.set_yscale('log')

    # Label the axes and provide a title
    ax.set_title("Some example Generations and their Accuracy and Loss")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(["1 Generation", "2 Generation", "3 Generation", "4 Generation", "Winner"], loc="upper left")
    axes = plt.gca()
    axes.set_ylim([0, 2])
    axes.set_xlim([0.5, 0.95])
    plt.gca().invert_yaxis()
    plt.show()

def scatterplot_zoom(file, yscale_log=False):
    x_label = "acc"
    y_label = "loss"
    xmin = 0.8
    xmax = 0.9
    ymax= 0.8
    ymin = 0.2
    # Create the plot object
    _, ax = plt.subplots()

    with open(file, "r") as f:
        data = json.load(f)

    for pop in data:
        x = []
        y = []
        if pop in ("2", "4", "8", "Winner"):
            for individum in data[pop]:
                try:
                    x.append(float(data[pop][individum]["acc"])) ## hier dürfte noch ein fehler sein
                    y.append(float(data[pop][individum]["loss"]))  ## hier dürfte noch ein feheler sein
                except:
                    print("error")

            # Plot the data, set the size (s), color and transparency (alpha)
            # of the points
            ax.scatter(x, y, s=60, alpha=0.7)

            if yscale_log == True:
                ax.set_yscale('log')

    # Label the axes and provide a title
    ax.set_title("Some example Generations and their Accuracy and Loss Zoomed in")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(["2 Generation", "4 Generation", "8 Generation", "Winner"], loc="upper left")
    axes = plt.gca()
    axes.set_xlim([xmin, xmax])
    axes.set_ylim([ymin, ymax])
    plt.gca().invert_yaxis()
    plt.show()


def plot_fitness(file):
    with open(file, "r") as f:
        data = json.load(f)

    title = "Fitness of Population"
    x_label = "Generations"
    y_label = "Fitness"
    acc_pop =[]

    for population in data:
        acc = 0
        anzahl = 0
        for individum in data[population]:
            acc += float(data[population][individum]["acc"])
            anzahl += 1
        acc = acc / anzahl
        acc_pop.append(acc)

    print(acc_pop)

    plt.plot(np.arange(len(acc_pop)), acc_pop)

    data = (np.arange(len(acc_pop)), acc_pop)

    #sns.lineplot(data=data)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

if __name__ == "__main__":




    save_file = "{}.{}.{}.json".format(datetime.datetime.now().year,
                                       datetime.datetime.now().month,
                                       datetime.datetime.now().day)
    save_file = "2019.8.14-2.json"

    plot_fitness(save_file)

