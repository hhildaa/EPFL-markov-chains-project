"""
For the simulations, you are free to fix parameters as you wish and explore. However, here are a
few guidelines and reasonable values: d = 3, N ranges between 100 and 1000, make about 100
experiments to compute the empirical average of the overlap.
Explore and compare the above algorithms and in particular:
• Investigate the average overlap as a function of time. Compare the convergence time of the
various algorithms (a meaningful measure of time is given by the number of steps of each
algorithm).

• Plot the average (over experiments) of limt→+∞ qN (t) as a function of b
a ∈ (0, 1]. Compare the limiting performance of different algorithms.

• Try to pinpoint the phase transition. Ideal phase transitions occur for N → +∞, so you will
necessarily find a smooth curve for finite N . By looking at the evolution of curves for many
N values, you may be able to get a rough indication of the phase transition point
"""

# TODO:
# For each ratio a/b, simulate per algorithm to what value of qn it converges, plot this as a fct of a/b;
# N is between 100 and 1000

# Chose:
# N = 100, 500, 1000
# Algo: Houdayer, Metropolis, Mixed Houdayer
# b/a in [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#        [10/990, 100/900, 200/800, 300/700, 40/600, 500/500]

import pickle

import numpy as np

from graph import generate_x_hat, generate_x, generate_adjacency_matrix
from metropolis import metropolis, houdayer
from phase_plots import theoretical_phase_transition
from utils import save_pickle, DATAPATH

DATA_PATH = './data/'

if __name__ == '__main__':
    a_list = None
    b_list = None
    n_0 = None
    algorithm = 'houdayer'
    rounds = 10

    for N in [100, 500, 1000]:

        if N == 100:
            a_list = np.array([70, 60, 57, 55, 54, 53, 52, 51, 50])
            b_list = np.array([30, 40, 43, 45, 46, 47, 48, 49, 50])
        elif N == 500:
            a_list = np.array([70, 60, 57, 55, 54, 53, 52, 51, 50]) * 5
            b_list = np.array([30, 40, 43, 45, 46, 47, 48, 49, 50]) * 5
        elif N == 1000:
            a_list = np.array([70, 60, 57, 55, 54, 53, 52, 51, 50]) * 10
            b_list = np.array([30, 40, 43, 45, 46, 47, 48, 49, 50]) * 10

        it_nums = [500, 500, 500, 500, 500, 500, 500, 500, 500]

        assert len(a_list) == len(b_list) == len(it_nums)
        assert [a_list[i]+b_list[i] for i in range(len(a_list))] == [N]*len(a_list)

        beta = 1
        avg_estimation_overlaps = []

        r_crit = theoretical_phase_transition(a_list[0], b_list[0])
        print(f'Critical phase transition expected at r_crit = {round(r_crit, 4)}.')

        for idx in range(0, len(a_list)):
            a = a_list[idx]
            b = b_list[idx]
            it_num = it_nums[idx]
            print(f'Current a = {a} Current b = {b}')
            print(f'Current r = {round(b/a, 4)}.')

            x_star = generate_x(N, 5)
            adj_matrix = generate_adjacency_matrix(x_star, a, b, N)

            estimation_overlap_over_runs = []
            estimation_overlap_over_runs_2 = []

            for experiment_run in range(rounds):
                x_hat_1 = generate_x_hat(N)
                if algorithm == "metropolis":
                    n_0 = 0
                    _, estimation_overlaps = \
                        metropolis(x_hat_1, adj_matrix, a, b, N, beta, x_star, it_num=it_num)

                elif algorithm == "houdayer":
                    n_0 = 2  # Normal houdayer
                    x_hat_2 = generate_x_hat(N)
                    estimation_overlaps, estimation_overlaps_2 = \
                        houdayer(x_hat_1, x_hat_2, adj_matrix, a, b, N, beta, x_star, n_0=n_0, it_num=it_num)
                    estimation_overlap_over_runs_2.append(estimation_overlaps_2[-1])

                estimation_overlap_over_runs.append(estimation_overlaps[-1])

                print(f'Iteration: {experiment_run + 1}/ {rounds} \t overlap: {round(estimation_overlaps[-1], 4)}')

            print(round(sum(estimation_overlap_over_runs) / len(estimation_overlap_over_runs), 4))
            if algorithm == "metropolis":
                avg_estimation_overlaps.append(round(sum(estimation_overlap_over_runs)
                                                     / len(estimation_overlap_over_runs), 4))
            elif algorithm == "houdayer":
                chain_1 = sum(estimation_overlap_over_runs) / len(estimation_overlap_over_runs)
                chain_2 = sum(estimation_overlap_over_runs_2) / len(estimation_overlap_over_runs_2)
                avg_estimation_overlaps.append(round(np.mean([chain_1, chain_2]), 4))

        b_div_a = np.array(b_list) / np.array(a_list)
        save_dict = {'b_div_a': b_div_a, 'N': N, 'algo': algorithm, 'avg_estimation_overlap': avg_estimation_overlaps}

        save_pickle(save_dict, DATAPATH, f'algo_{algorithm}_N_{N}_n0_{n_0}' + '.pickle')
