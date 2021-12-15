import numpy as np
from matplotlib import pyplot as plt

from utils import load_pickle


def theoretical_phase_transition(a, b):
    d = 1/2 * (a + b)
    b_crit = d - np.sqrt(d)
    a_crit = d + np.sqrt(d)
    r_crit = b_crit/a_crit
    return r_crit


def plot_phase_transition_metropolis(x, ys, ns, r_crits, algorithm):
    colors = ["blue", "orange", "green"]
    plt.figure()
    x = np.insert(x, 0, 0)
    ys = np.array(ys)
    ys = np.insert(ys, 0, 1, axis=1)
    for idx in range(len(ns)):
        plt.plot(x, ys[idx], label=str(ns[idx]), c=colors[idx % len(ns)])
        plt.vlines(r_crits[idx], ymin=0, ymax=1,
                   colors=colors[idx % len(ns)], linestyles="dashed")
    plt.xlabel("b/a")
    plt.ylabel("limit of the empirical overlap q_n")
    plt.legend()
    plt.savefig(f"plots/{algorithm}_phase_transition.jpg")
    plt.show()


def metropolis_phase_viz(r_crits, ns):
    # Metropolis
    # it_nums = [5000, 5000, 6000, 6000, 6000, 6000, 6000, 1000, 1000]
    metropolis_100 = load_pickle("data/", "algo_metropolis_N_100.pickle")
    metropolis_500 = load_pickle("data/", "algo_metropolis_N_500.pickle")
    metropolis_1000 = load_pickle("data/", "algo_metropolis_N_1000.pickle")

    y_metropolis_100 = metropolis_100['avg_estimation_overlap']
    y_metropolis_500 = metropolis_500['avg_estimation_overlap']
    y_metropolis_1000 = metropolis_1000['avg_estimation_overlap']

    ys_metropolis = [y_metropolis_100, y_metropolis_500, y_metropolis_1000]

    plot_phase_transition_metropolis(metropolis_100['b_div_a'], ys_metropolis, ns, r_crits, algorithm="metropolis")


def houdayer_viz(r_crits, ns):
    # it_nums = [500, 500, 500, 500, 500, 500, 500, 500, 500]
    houdayer_100 = load_pickle("data/", "algo_houdayer_N_100_n0_2.pickle")
    houdayer_500 = load_pickle("data/", "algo_houdayer_N_500.pickle")
    houdayer_1000 = load_pickle("data/", "algo_houdayer_N_1000.pickle")

    y_houdayer_100 = houdayer_100['avg_estimation_overlap']
    y_houdayer_500 = houdayer_500['avg_estimation_overlap']
    y_houdayer_1000 = houdayer_1000['avg_estimation_overlap']

    ys_metropolis = [y_houdayer_100, y_houdayer_500, y_houdayer_1000]

    plot_phase_transition_metropolis(houdayer_100['b_div_a'], ys_metropolis, ns, r_crits, algorithm="houdayer_n0_2")


if __name__ == '__main__':
    r_crit_100 = theoretical_phase_transition(40, 60)
    r_crit_500 = theoretical_phase_transition(200, 300)
    r_crit_1000 = theoretical_phase_transition(400, 600)
    r_critical = [r_crit_100, r_crit_500, r_crit_1000]
    n_s = [100, 500, 1000]

    metropolis_phase_viz(r_critical, n_s)

    houdayer_viz(r_critical, n_s)

