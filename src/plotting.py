
import json
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
from scipy import stats

def plot_winner():
    with open("./data.json", "r") as f:
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

def plot_all():
    with open("./data.json", "r") as f:
        data = json.load(f)

    for pop in data:
        x = []
        y = []
        for individum in data[pop]:
            try:
                x.append(data[pop][individum]["acc"])    ## hier dürfte noch ein fehler sein
                y.append(data[pop][individum]["loss"])   ## hier dürfte noch ein feheler sein
            except:
                print(pop,individum)
        plt.scatter(x, y, s=80, marker="+")
        plt.xlabel('acc', fontsize=18)
        plt.ylabel('loss', fontsize=16)
        plt.gca().invert_yaxis()
        plt.show()


def plot_histogram(names,werteliste):
    num_bins = 50 #bins sind Balken
    kde = stats.gaussian_kde(werteliste)

    #y = mlab.normpdf(num_bins, np.mean(werteliste), np.std(werteliste))
    n, bins, patches = plt.hist(werteliste, num_bins, density=True, facecolor='blue', alpha=0.5)


    plt.plot(bins, kde, 'r--')
    plt.xlabel('-')
    plt.ylabel('Probability')
    plt.title(names)
    plt.show()

def plot_histogram_all():
    with open("./data.json", "r") as f:
        data = json.load(f)

    learningrate=[]
    batchsize=[]
    dropout=[]
    epoch =[]

    anzahl = 0
    for individum in data["Winner"]:
        learningrate.append(data["Winner"][individum]["learningrate"])
        batchsize.append(data["Winner"][individum]["batchsize"])
        dropout.append(data["Winner"][individum]["dropout"])
        epoch.append(data["Winner"][individum]["epoch"])
        anzahl += 1

    print('learningrate mean=%.5f stdv=%.5f' % (np.mean(learningrate), np.std(learningrate)))
    plot_histogram("learningrate",learningrate)
    print('batchsize mean=%.5f stdv=%.5f' % (np.mean(batchsize), np.std(batchsize)))
    plot_histogram("batchsize",batchsize)
    print('dropout mean=%.5f stdv=%.5f' % (np.mean(dropout), np.std(dropout)))
    plot_histogram("dropout",dropout)
    print('epoch mean=%.5f stdv=%.5f' % (np.mean(epoch), np.std(epoch)))
    plot_histogram("epoch",epoch)


def scatterplot_log():

    title ="loss over acc"
    x_label = "acc"
    y_label = "loss"
    yscale_log = True


    # Create the plot object
    _, ax = plt.subplots()

    with open("./data.json", "r") as f:
        data = json.load(f)

    for pop in data:
        x = []
        y = []
        i= 0
        if pop in ("2","4","8","Winner"):
            for individum in data[pop]:
                try:
                    x.append(data[pop][individum]["acc"])  ## hier dürfte noch ein fehler sein
                    y.append(data[pop][individum]["loss"])  ## hier dürfte noch ein feheler sein
                except:
                    print("error")

            #cmap = mpl.cm.autumn
            #color = cmap(i / float(4))

            #color = colorarray[i]
            # Plot the data, set the size (s), color and transparency (alpha)
            # of the points
            ax.scatter(x, y, s=40, alpha=0.5)

            if yscale_log == True:
                ax.set_yscale('log')

            # Label the axes and provide a title
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend(["2 generation","4 generation","8 generation","Winner"],loc = "upper left")
        plt.gca().invert_yaxis()

        plt.show


def scatterplot():

    title ="loss over acc"
    x_label = "acc"
    y_label = "loss"
    yscale_log = True

    xmin = 0.8
    xmax = 0.9
    ymin = 0.8
    ymax = 0.2
    colorarray=["r","b","g","b"]

    # Create the plot object
    _, ax = plt.subplots()

    with open("./data.json", "r") as f:
        data = json.load(f)

    for pop in data:
        x = []
        y = []
        i= 0
        if pop in ("2","4","8","Winner"):
            for individum in data[pop]:
                try:
                    x.append(data[pop][individum]["acc"])  ## hier dürfte noch ein fehler sein
                    y.append(data[pop][individum]["loss"])  ## hier dürfte noch ein feheler sein
                except:
                    print("error")

            #cmap = mpl.cm.autumn
            #color = cmap(i / float(4))

            #color = colorarray[i]
            # Plot the data, set the size (s), color and transparency (alpha)
            # of the points
            ax.scatter(x, y, s=40, alpha=0.5)

            if yscale_log == True:
                ax.set_yscale('log')

            # Label the axes and provide a title
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend(["2 generation","4 generation","8 generation","Winner"],loc = "upper left")
        plt.gca().invert_yaxis()

        plt.show

    yscale_log = False
    # Create the plot object
    _, ax = plt.subplots()

    with open("./data.json", "r") as f:
        data = json.load(f)

    for pop in data:
        x = []
        y = []
        i = 0
        if pop in ("2", "4", "8", "Winner"):
            for individum in data[pop]:
                try:
                    x.append(data[pop][individum]["acc"])  ## hier dürfte noch ein fehler sein
                    y.append(data[pop][individum]["loss"])  ## hier dürfte noch ein feheler sein
                except:
                    print("error")

            # cmap = mpl.cm.autumn
            # color = cmap(i / float(4))

            # color = colorarray[i]
            # Plot the data, set the size (s), color and transparency (alpha)
            # of the points
            ax.scatter(x, y, s=60, alpha=0.7)

            if yscale_log == True:
                ax.set_yscale('log')

    # Label the axes and provide a title
    ax.set_title("Zoomed in")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(["2 generation", "4 generation", "8 generation", "Winner"], loc="upper left")
    plt.gca().invert_yaxis()
    axes = plt.gca()
    axes.set_xlim([xmin, xmax])
    axes.set_ylim([ymin, ymax])

    plt.show()




if __name__ == "__main__":

    scatterplot()


    #plot_all()
    plot_winner()
    plot_histogram_all()


