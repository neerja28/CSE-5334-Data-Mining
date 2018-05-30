#!/usr/bin/python3

from matplotlib import pyplot as plt


def display_graph(ylabels):
    xlabels = []
    for temp in range(len(ylabels)):
        xlabels.append(temp)

    plt.plot(xlabels, ylabels)
    plt.show()
