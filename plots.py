import numpy as np
from matplotlib import pyplot as plt
import networkx as nx


def visualize_graph(adj, color, n):
    # A: Fixed bug where you were using edges variable but should have been using adj

    G = nx.Graph(adj - np.eye(n))  # had edges variable here
    color_map = np.where(color == 1, 'blue', 'red')
    nx.draw(G, node_size=50, node_color=color_map)


def aggregate_acceptance_rates(acceptances_list):
    aggregates = np.array_split(np.array(acceptances_list), 20)
    total_iteration = [x.shape[0] for x in aggregates]
    aggregates = [np.sum(x) for x in aggregates]
    return aggregates, total_iteration


def plot_acceptance_rates(acceptances_list):
    # Aggregate the ones and zeros over time
    aggregate, total_its = aggregate_acceptance_rates(acceptances_list)
    x = list(range(len(aggregate)))
    plt.figure()
    plt.bar(x, total_its, width=0.5)
    plt.bar(x, aggregate, width=0.5)
    plt.xlabel("Iterations aggregated")
    plt.ylabel("Proportion of accepted moves")
    plt.legend(["total proposals", "accepted proposals"], loc='best')
    plt.show()


def plot_estimation_overlaps_over_iterations(estimation_overlaps):
    # For better seeing the development of the plot,
    # Visualise only every tenth iteration
    plt.figure()
    for elem in estimation_overlaps:
        plt.plot(elem)
    plt.xlabel("Iterations")
    plt.ylabel("Estimation overlap")
    plt.title("Estimation overlap over iterations")
    plt.show()
