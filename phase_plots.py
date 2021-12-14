import numpy as np
from matplotlib import pyplot as plt

from utils import load_pickle


def theoretical_phase_transition(a, b):
    d = 1/2 * (a + b)
    b_crit = d - np.sqrt(d)
    a_crit = d + np.sqrt(d)
    r_crit = b_crit/a_crit
    return r_crit


def plot_phase_transition_metropolis(x, ys, ns, r_crits):
    colors = ["blue", "orange", "green"]
    plt.figure()
    for idx in range(len(ns)):
        plt.plot(x, ys[idx], label=str(ns[idx]), c=colors[idx % len(ns)])
        plt.vlines(r_crits[idx], ymin=0, ymax=1,
                   colors=colors[idx % len(ns)], linestyles="dashed")
    plt.xlabel("b/a")
    plt.ylabel("limit of the empirical overlap q_n")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # Metropolis
    metropolis_100 = load_pickle("data/", "algo_metropolis_N_100.pickle")
    metropolis_1000 = load_pickle("data/", "algo_metropolis_N_1000.pickle")

    y_metropolis_100 = metropolis_100['avg_estimation_overlap']
    r_crit_metropolis_100 = theoretical_phase_transition(40, 60)

    y_metropolis_500 = [1, 1, 1, 1, 1, 1, 1, 0, 0]
    r_crit_metropolis_500 = theoretical_phase_transition(200, 300)

    y_metropolis_1000 = metropolis_1000['avg_estimation_overlap']
    r_crit_metropolis_1000 = theoretical_phase_transition(400, 600)

    r_crits_metropolis = [r_crit_metropolis_100, r_crit_metropolis_500, r_crit_metropolis_1000]
    ys_metropolis = [y_metropolis_100, y_metropolis_500, y_metropolis_1000]
    ns_metropolis = [100, 500, 1000]

    plot_phase_transition_metropolis(metropolis_100['b_div_a'], ys_metropolis, ns_metropolis, r_crits_metropolis)
