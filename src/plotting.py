
import json
import matplotlib.pyplot as plt
import numpy as np

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
            x.append(data[pop][gen]["acc"])    ## hier dürfte noch ein fehler sein
            y.append(data[pop][gen]["loss"])   ## hier dürfte noch ein feheler sein
        plt.scatter(x, y, s=80, marker="+")
        plt.xlabel('acc', fontsize=18)
        plt.ylabel('loss', fontsize=16)
        plt.gca().invert_yaxis()
        plt.show(num=gen)

def plot_normalverteilung(names,werteliste):
    num_bins = 10 #bins sind Balken
    plt.hist(werteliste, num_bins, density=True, facecolor='blue', alpha=0.5)

    # add a 'best fit' line
    #y = norm(bins, np.mean(learningrate), np.std(learningrate))

    #plt.plot(bins, y, 'r--')
    plt.xlabel('Smarts')
    plt.ylabel('Probability')
    plt.title(names)

    # Tweak spacing to prevent clipping of ylabel
    plt.subplots_adjust(left=0.15)
    plt.show()


def plot_normalverteilung_all():
    with open("./data.json", "r") as f:
        data = json.load(f)

    learningrate=[]
    batchsize=[]
    dropout=[]
    epoch =[]

    anzahl = 0
    for gen in data["Winner"]:
        learningrate.append(data["Winner"][gen]["learningrate"])
        batchsize.append(data["Winner"][gen]["batchsize"])
        dropout.append(data["Winner"][gen]["dropout"])
        epoch.append(data["Winner"][gen]["epoch"])
        anzahl += 1

    print('learningrate mean=%.5f stdv=%.5f' % (np.mean(learningrate), np.std(learningrate)))
    plot_normalverteilung("learningrate",learningrate)
    print('batchsize mean=%.5f stdv=%.5f' % (np.mean(batchsize), np.std(batchsize)))
    plot_normalverteilung("batchsize",batchsize)
    print('dropout mean=%.5f stdv=%.5f' % (np.mean(dropout), np.std(dropout)))
    plot_normalverteilung("dropout",dropout)
    print('epoch mean=%.5f stdv=%.5f' % (np.mean(epoch), np.std(epoch)))
    plot_normalverteilung("epoch",epoch)



if __name__ == "__main__":
    plot_winner()
    plot_normalverteilung_all()


