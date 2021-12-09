import numpy as np
from matplotlib import pyplot as plt


def aggregate_acceptance_rates(acceptances_list):
    aggregates = np.array_split(np.array(acceptances_list), 10)
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


if __name__ == '__main__':
    acceptances_lists = []
    for i in range(1):
        acceptances_lists.append(np.random.randint(2, size=10000).tolist())
    plot_acceptance_rates(acceptances_lists)
