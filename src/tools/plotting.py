# coding=utf-8
import json
import matplotlib.mlab as mlab
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib as mpl
from scipy import stats

import datetime
import os

sns.set(color_codes=True)


def scatterplot(dir_path, save_file, yscale_log=False, save=False):
    x_label = "acc"
    y_label = "loss"

    # Create the plot object
    _, ax = plt.subplots()
    save_file = os.path.join(dir_path, "../data/",save_file)
    with open(save_file, "r") as f:
        data = json.load(f)

    for pop in data["generation"]:
        x = []
        y = []
        if pop in ("0", "2", "Winner"):
            for individum in data["generation"][pop]:
                try:
                    x.append(float(data["generation"][pop][individum]["acc"])) ## hier dürfte noch ein fehler sein
                    y.append(float(data["generation"][pop][individum]["loss"]))  ## hier dürfte noch ein feheler sein
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

    if save:
        filename = os.path.join(dir_path, "../data/",save_file)
        filename = filename[:-5]
        filename = filename + "_scaterplot.pdf"
        filename = os.path.abspath(os.path.realpath(filename))
        plt.savefig(filename)

def joint_plot(dir_path, save_file, save=False):
    save_file = os.path.join(dir_path, "../data/",save_file)
    with open(save_file, "r") as f:
        data = json.load(f)
    learningrate = []
    dropout = []
    epoch = []
    batchsize = []
    optimizer = []
    acc = []
    loss = []
    variables = []
    xmin = 0.8
    xmax = 0.9
    ymax= 0.8
    ymin = 0.2
    for i in data["generation"]["Winner"]:
        if float(data["generation"]["Winner"][i]["acc"]) > float(data["generation"]["Winner"]["0"]["acc"]) * 0.8:
            learningrate.append(float(data["generation"]["Winner"][i]["learningrate"]))
            dropout.append(float(data["generation"]["Winner"][i]["dropout"]))
            epoch.append(float(data["generation"]["Winner"][i]["epoch"]))
            batchsize.append(float(data["generation"]["Winner"][i]["batchsize"]))
            optimizer.append(float(data["generation"]["Winner"][i]["optimizer"]))
            acc.append(float(data["generation"]["Winner"][i]["acc"]))
            loss.append(float(data["generation"]["Winner"][i]["loss"]))
            variables.append(float(data["generation"]["Winner"][i]["variables"]))
        else:
            pass
  
    auswertungsdaten = {"learningrate":learningrate,
                        "dropout":dropout,
                        "epoch":epoch,
                        "batchsize":batchsize,
                        "optimizer":optimizer,
                        "acc": acc,
                        "loss":loss,
                        "variables": variables
                        }

    df = pd.DataFrame(auswertungsdaten,columns = ["learningrate","dropout","epoch","batchsize","optimizer"
                            ,"acc","loss","variables"])

    if save:
        filename = os.path.join(dir_path, "../data/",save_file)
        filename = filename[:-5]
        g = (sns.jointplot("acc", "learningrate", data=df)
        .plot_joint(sns.kdeplot, n_levels=6))
        plt.title("learningrate")
        filename = filename + "_jointplot_learningrate.pdf"
        filename = os.path.abspath(os.path.realpath(filename))
        plt.savefig(filename)
        g = (sns.jointplot("acc", "dropout", data=df)
        .plot_joint(sns.kdeplot, n_levels=6))
        plt.title("dropout")
        filename = filename + "_jointplot_dropout.pdf"
        filename = os.path.abspath(os.path.realpath(filename))
        plt.savefig(filename)
        g = (sns.jointplot("acc", "epoch", data=df)
        .plot_joint(sns.kdeplot, n_levels=6))
        plt.title("epoch")
        filename = filename + "_jointplot_epoch.pdf"
        filename = os.path.abspath(os.path.realpath(filename))
        plt.savefig(filename)
        g = (sns.jointplot("acc", "batchsize", data=df)
        .plot_joint(sns.kdeplot, n_levels=6))
        plt.title("batchsize")
        filename = filename + "_jointplot_batchsize.pdf"
        filename = os.path.abspath(os.path.realpath(filename))
        plt.savefig(filename)
        g = (sns.jointplot("acc", "optimizer", data=df, kind="kde")
        .plot_joint(sns.kdeplot, n_levels=6))
        plt.title("optimizer")
        filename = filename + "_jointplot_optimizer_kde.pdf"
        filename = os.path.abspath(os.path.realpath(filename))
        plt.savefig(filename)
        g = (sns.jointplot("acc", "optimizer", data=df)
        .plot_joint(sns.kdeplot, n_levels=6))
        plt.title("optimizer")
        filename = filename + "_jointplot_optimizer.pdf"
        filename = os.path.abspath(os.path.realpath(filename))
        plt.savefig(filename)
        g = (sns.jointplot("acc", "variables", data=df)
        .plot_joint(sns.kdeplot, n_levels=6))
        plt.title("variables")
        filename = filename + "_jointplot_variables.pdf"
        filename = os.path.abspath(os.path.realpath(filename))
        plt.savefig(filename)
    else:
        g = (sns.jointplot("acc", "learningrate", data=df)
        .plot_joint(sns.kdeplot, n_levels=6))
        g = (sns.jointplot("acc", "dropout", data=df)
        .plot_joint(sns.kdeplot, n_levels=6))
        g = (sns.jointplot("acc", "epoch", data=df)
        .plot_joint(sns.kdeplot, n_levels=6))
        g = (sns.jointplot("acc", "batchsize", data=df)
        .plot_joint(sns.kdeplot, n_levels=6))
        g = (sns.jointplot("acc", "optimizer", data=df, kind="kde")
        .plot_joint(sns.kdeplot, n_levels=6))
        g = (sns.jointplot("acc", "optimizer", data=df)
        .plot_joint(sns.kdeplot, n_levels=6))
        g = (sns.jointplot("acc", "variables", data=df)
        .plot_joint(sns.kdeplot, n_levels=6))
        plt.show()




if __name__ == "__main__":

    save_file = "{}.{}.{}.json".format(datetime.datetime.now().year,
                                       datetime.datetime.now().month,
                                       datetime.datetime.now().day)
    save_file = "ergebnisse_hyper.json"
    save_file = "2019.11.7.json"
    dir_path = os.path.dirname(os.path.realpath(__file__))
    scatterplot(dir_path= dir_path, save_file=save_file, save = True)
    joint_plot(dir_path= dir_path, save_file=save_file, save = True)




def scatterplot_zoom(dir_path, save_file, yscale_log=False, save=False):
    x_label = "acc"
    y_label = "loss"
    xmin = 0.8
    xmax = 0.9
    ymax= 0.8
    ymin = 0.2
    # Create the plot object
    _, ax = plt.subplots()
    save_file = os.path.join(dir_path, "../data/",save_file)
    with open(save_file, "r") as f:
        data = json.load(f)

    for pop in data["generation"]:
        x = []
        y = []
        if pop in ("0", "2", "Winner"):
            for individum in data["generation"][pop]:
                try:
                    x.append(float(data["generation"][pop][individum]["acc"])) ## hier dürfte noch ein fehler sein
                    y.append(float(data["generation"][pop][individum]["loss"]))  ## hier dürfte noch ein feheler sein
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
    ax.legend(["0 Generation", "2 Generation", "Winner"], loc="upper left")
    axes = plt.gca()
    axes.set_xlim([xmin, xmax])
    axes.set_ylim([ymin, ymax])
    plt.gca().invert_yaxis()
    plt.show()

    if save:
        filename = os.path.join(dir_path, "../data/",save_file)
        filename = filename[:-5]
        filename = filename + "_scaterplot_zoom.pdf"
        filename = os.path.abspath(os.path.realpath(filename))
        plt.savefig(filename)

def plot_fitness(dir_path, save_file, save=False):
    save_file = os.path.join(dir_path, "../data/",save_file)
    with open(save_file, "r") as f:
        data = json.load(f)

    title = "Fitness of Population"
    x_label = "Generations"
    y_label = "Fitness"
    acc_pop =[]

    for population in data["generation"]:
        acc = 0
        anzahl = 0
        for individum in data["generation"][population]:
            acc += float(data["generation"][population][individum]["acc"])
            anzahl += 1
        acc = acc / anzahl
        acc_pop.append(acc)
    plt.plot(np.arange(len(acc_pop)), acc_pop)
    data = (np.arange(len(acc_pop)), acc_pop)

    #sns.lineplot(data=data)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

    if save:
        filename = os.path.join(dir_path, "../data/",save_file)
        filename = filename[:-5]
        filename = filename + "_fitness.pdf"
        filename = os.path.abspath(os.path.realpath(filename))
        plt.savefig(filename)



def plot_winner(save_file):
    with open(save_file, "r") as f:
        data = json.load(f)
    x = []
    y = []
    xmin = 0.8
    xmax = 0.9
    ymin = 0.8
    ymax = 0.2

    for gen in data["generation"]["Winner"]:
        x.append(data["generation"]["Winner"][gen]["acc"])
        y.append(data["generation"]["Winner"][gen]["loss"])

    plt.scatter(x, y, s=80, marker="+")
    plt.xlabel('acc', fontsize=18)
    plt.ylabel('loss', fontsize=16)
    plt.gca().invert_yaxis(),
    axes = plt.gca()
    axes.set_xlim([xmin, xmax])
    axes.set_ylim([ymin, ymax])
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


def box_plot(save_file):
    with open(save_file, "r") as f:
        data = json.load(f)
    learningrate = []
    dropout = []
    epoch = []
    batchsize = []
    optimizer = []
    acc = []
    loss = []
    variables = []

    for i in data["generation"]["Winner"]:
        learningrate.append(float(data["generation"]["Winner"][i]["learningrate"]))
        dropout.append(float(data["generation"]["Winner"][i]["dropout"]))
        epoch.append(float(data["generation"]["Winner"][i]["epoch"]))
        batchsize.append(float(data["generation"]["Winner"][i]["batchsize"]))
        optimizer.append(float(data["generation"]["Winner"][i]["optimizer"]))
        acc.append(float(data["generation"]["Winner"][i]["acc"]))
        loss.append(float(data["generation"]["Winner"][i]["loss"]))
        variables.append(float(data["generation"]["Winner"][i]["variables"]))
  
    auswertungsdaten = {"learningrate":learningrate,
                        "dropout":dropout,
                        "epoch":epoch,
                        "batchsize":batchsize,
                        "optimizer":optimizer,
                        "acc": acc,
                        "loss":loss,
                        "variables": variables
                        }

    df = pd.DataFrame(auswertungsdaten,columns = ["learningrate","dropout","epoch","batchsize","optimizer"
                            ,"acc","loss","variables"])
    stats_df = pd.DataFrame(auswertungsdaten,columns = ["learningrate"])
    g = (sns.boxplot(data=stats_df)
    .plot_joint(sns.kdeplot, n_levels=6))

    plt.show()