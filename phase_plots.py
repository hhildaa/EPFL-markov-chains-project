import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from utils import load_pickle


def theoretical_phase_transition(a, b):
    d = 1 / 2 * (a + b)
    b_crit = d - np.sqrt(d)
    a_crit = d + np.sqrt(d)
    r_crit = b_crit / a_crit
    return r_crit


def plot_N_against_each_other(r_crit, n):
    # houdayer_n0_2 = load_pickle("data/", f"algo_houdayer_N_{n}_n0_2.pickle")
    houdayer_n0_5 = load_pickle("data/", f"algo_houdayer_N_{n}_n0_5.pickle")
    houdayer_n0_10 = load_pickle("data/", f"algo_houdayer_N_{n}_n0_10.pickle")
    houdayer_n0_15 = load_pickle("data/", f"algo_houdayer_N_{n}_n0_15.pickle")
    houdayer_n0_30 = load_pickle("data/", f"algo_houdayer_N_{n}_n0_30.pickle")

    # y_houdayer_n0_2 = houdayer_n0_2['avg_estimation_overlap']
    y_houdayer_n0_5 = houdayer_n0_5['avg_estimation_overlap']
    y_houdayer_n0_10 = houdayer_n0_10['avg_estimation_overlap']
    y_houdayer_n0_15 = houdayer_n0_15['avg_estimation_overlap']
    y_houdayer_n0_30 = houdayer_n0_30['avg_estimation_overlap']

    ys = [y_houdayer_n0_5, y_houdayer_n0_10,
          y_houdayer_n0_15, y_houdayer_n0_30]
    n0s = [5, 10, 15, 30]
    x = houdayer_n0_5['b_div_a']

    # colors = ["blue", "orange", "green"]
    plt.figure()
    x = np.insert(x, 0, 0)
    ys = np.array(ys)
    ys = np.insert(ys, 0, 1, axis=1)
    for idx in range(len(ys)):
        plt.plot(x, ys[idx], label=str(n0s[idx]))
        plt.vlines(r_crit, ymin=0, ymax=1, linestyles="dashed")
    plt.xlabel("b/a")
    plt.ylabel("limit of the empirical overlap q_n")
    plt.legend()
    plt.savefig(f"plots/houdayer_phase_transition_n_{n}.jpg")
    plt.show()


def get_data_per_degree(x, ys, ns):
    dfs = []
    for i in range(len(ns)):
        temp = (pd.DataFrame(ys[:, :, i], columns=x).stack().reset_index()
                .drop(columns="level_0")
                )
        temp.insert(2, "N", ns[i])
        dfs.append(temp)
    data = pd.concat(dfs).rename(columns={"level_1": "x", 0: "y"})
    data = data.reset_index().drop(columns="index")
    return data


def get_data_in_plot_format(x, ys):
    ns = [100, 500, 1000]
    data = get_data_per_degree(x, ys, ns)
    return data


def plot_phase_transition_metropolis(x, ys, r_crits, algorithm, degree):
    palette = sns.color_palette("mako_r", 3)
    ys = np.array(ys).T

    data = get_data_in_plot_format(x, ys)

    sns.lineplot(data=data, x="x", y="y", hue="N", palette=palette)

    for i in range(len(r_crits)):
        plt.vlines(r_crits[i], ymin=0, ymax=1, linestyles="dashed", colors=palette[i])

    plt.xlabel("b/a")
    plt.ylabel("limit of the empirical overlap q_n")
    plt.savefig(f"plots/{algorithm}_phase_transition_degree_{degree}.jpg")
    plt.show()


def metropolis_phase_viz(r_crits, degree):
    # Metropolis
    metropolis_100 = load_pickle("data/", f"algo_metropolis_N_100_degree_{degree}.pickle")
    metropolis_500 = load_pickle("data/", f"algo_metropolis_N_500_degree_{degree}.pickle")
    metropolis_1000 = load_pickle("data/", f"algo_metropolis_N_1000_degree_{degree}.pickle")

    ys_metropolis = [metropolis_100['avg_estimation_overlap'],
                     metropolis_500['avg_estimation_overlap'],
                     metropolis_1000['avg_estimation_overlap']]

    plot_phase_transition_metropolis(metropolis_100['b_div_a'],
                                     ys_metropolis,
                                     r_crits,
                                     algorithm="metropolis",
                                     degree=degree)


def houdayer_viz(r_crits, ns):
    houdayer_100 = load_pickle("data/", "algo_houdayer_N_100_n0_2.pickle")
    houdayer_500 = load_pickle("data/", "algo_houdayer_N_500_n0_2.pickle")
    houdayer_1000 = load_pickle("data/", "algo_houdayer_N_1000_n0_2.pickle")

    y_houdayer_100 = houdayer_100['avg_estimation_overlap']
    y_houdayer_500 = houdayer_500['avg_estimation_overlap']
    y_houdayer_1000 = houdayer_1000['avg_estimation_overlap']

    ys_metropolis = [y_houdayer_100, y_houdayer_500, y_houdayer_1000]

    plot_phase_transition_metropolis(houdayer_100['b_div_a'],
                                     ys_metropolis,
                                     ns,
                                     r_crits,
                                     algorithm="houdayer",
                                     n0=2)


def houdayer_mix_viz(r_crits, ns, n0):
    # it_nums = [500, 500, 500, 500, 500, 500, 500, 500, 500]
    houdayer_100 = load_pickle("data/", f"algo_houdayer_N_100_n0_{n0}.pickle")
    houdayer_500 = load_pickle("data/", f"algo_houdayer_N_500_n0_{n0}.pickle")
    houdayer_1000 = load_pickle("data/", f"algo_houdayer_N_1000_n0_{n0}.pickle")

    y_houdayer_100 = houdayer_100['avg_estimation_overlap']
    y_houdayer_500 = houdayer_500['avg_estimation_overlap']
    y_houdayer_1000 = houdayer_1000['avg_estimation_overlap']

    ys_metropolis = [y_houdayer_100, y_houdayer_500, y_houdayer_1000]

    plot_phase_transition_metropolis(houdayer_100['b_div_a'],
                                     ys_metropolis, ns, r_crits,
                                     algorithm="houdayer",
                                     n0=n0)


if __name__ == '__main__':
    r_crit_100 = theoretical_phase_transition(40, 60)
    r_crit_500 = theoretical_phase_transition(200, 300)
    r_crit_1000 = theoretical_phase_transition(400, 600)
    r_critical = [r_crit_100, r_crit_500, r_crit_1000]
    n_0 = 10
    n_s = [100, 500, 1000]
    metropolis_phase_viz(r_critical, degree=5)
    metropolis_phase_viz(r_critical, degree=50)

    plot_N_against_each_other(r_crit_100, 100)
    plot_N_against_each_other(r_crit_500, 500)
    plot_N_against_each_other(r_crit_1000, 1000)

    houdayer_mix_viz(r_critical, n_s, n0=5)
    houdayer_mix_viz(r_critical, n_s, n0=10)
    houdayer_mix_viz(r_critical, n_s, n0=15)
    houdayer_mix_viz(r_critical, n_s, n0=30)

    houdayer_viz(r_critical, n_s)
