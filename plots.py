import numpy as np
from matplotlib import pyplot as plt
import networkx as nx


def visualize_graph(adj, color, n):
    # A: Fixed bug where you were using edges variable but should have been using adj

    G = nx.Graph(adj - np.eye(n))  # had edges variable here
    color_map = np.where(color == 1, 'blue', 'red')
    nx.draw(G, node_size=50, node_color=color_map)


def aggregate_acceptance_rates(acceptances_list):
    aggregates = np.array_split(np.array(acceptances_list), 30)
    total_iteration = [x.shape[0] for x in aggregates]
    aggregates = [np.sum(x) for x in aggregates]
    return aggregates, total_iteration


def plot_acceptance_rates(acceptances_lists):
    aggregates = []
    total_iterations = []
    for acc_list in acceptances_lists:
        aggregate, total_its = aggregate_acceptance_rates(acc_list)
        aggregates.append(aggregate)
        total_iterations.append(total_its)
    x = list(range(len(aggregates[0])))

    for i in range(len(acceptances_lists)):
        plt.figure()
        plt.bar(x, total_iterations[i], width=0.5)
        plt.bar(x, aggregates[i], width=0.5)
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
