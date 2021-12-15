from graph import *
from metropolis import *
from plots import plot_acceptance_rates, visualize_graph, plot_estimation_overlaps_over_iterations
import pickle
from utils import DATAPATH, save_pickle
import time


def test(a, b, N, it_num, beta_0, n_0, sim_num, algo, beta_n=None, save=True):
    x_star = generate_x(N, 5)
    adj_matrix = generate_adjacency_matrix(x_star, a, b, N)
    random_avg = check_random_estimates(a, b, N, x_star, verbose=False)

    estimation_overlap_over_runs_1 = []
    estimation_overlap_over_runs_2 = []

    estimation_overlaps_first_perfect_1 = []
    estimation_overlaps_first_perfect_2 = []

    acceptance_rate_list = None

    print(f"=== a: {a}  b: {b}  N: {N} algo: {algo} iterations per run: {it_num} ===")

    if algo == 'houdayer':
        for i in range(sim_num):
            # starting position: random assignment of clusters
            x_hat_1 = generate_x_hat(N)
            x_hat_2 = generate_x_hat(N)

            estimation_overlaps_1, estimation_overlaps_2 = \
                houdayer(x_hat_1, x_hat_2, adj_matrix, a, b, N, beta_0, x_star, n_0=n_0, it_num=it_num)

            estimation_overlap_over_runs_1.append(estimation_overlaps_1)
            estimation_overlap_over_runs_2.append(estimation_overlaps_2)

            if estimation_overlaps_1.count(1) != 0:
                estimation_overlaps_first_perfect_1.append(estimation_overlaps_1.index(1))
                estimation_overlaps_first_perfect_2.append(estimation_overlaps_2.index(1))
            else:
                estimation_overlaps_first_perfect_1.append('nan')
                estimation_overlaps_first_perfect_2.append('nan')

            print(f'Iteration: {i + 1}/{sim_num} \t overlap: {round(estimation_overlaps_1[-1], 4)} \
                first perfect: {estimation_overlaps_first_perfect_1[-1]}, {estimation_overlaps_first_perfect_2[-1]}')

    else:
        for i in range(sim_num):
            # starting position: random assignment of clusters
            x_hat_1 = generate_x_hat(N)

            _, estimation_overlaps_1 = \
                metropolis(x_hat_1, adj_matrix, a, b, N, beta_0, x_star, beta_n=beta_n, it_num=it_num)

            estimation_overlap_over_runs_1.append(estimation_overlaps_1)

            if estimation_overlaps_1.count(1) != 0:
                estimation_overlaps_first_perfect_1.append(estimation_overlaps_1.index(1))
            else:
                estimation_overlaps_first_perfect_1.append('nan')

            print(f'Iteration: {i + 1}/{sim_num} \t overlap: {round(estimation_overlaps_1[-1], 4)}\
                first perfect: {estimation_overlaps_first_perfect_1[-1]}')

    if save:
        save_dict = {'a': a, 'b': b, 'N': N, 'it_num': it_num, 'beta_0': beta_0,'beta_n': beta_n, 'n_0': n_0, 'sim_num': sim_num, 'algo': algo,
                     'estimation_overlap_over_runs_1': estimation_overlap_over_runs_1,
                     'estimation_overlap_over_runs_2': estimation_overlap_over_runs_2,
                     'first_perfect_1': estimation_overlaps_first_perfect_1,
                     'first_perfect_2': estimation_overlaps_first_perfect_2}

        save_pickle(save_dict, DATAPATH, f'a_{a}_b_{b}_algo_{algo}_beta0_{beta_0}_betaN_{beta_n}_n0_{n_0}' + '.pickle')


if __name__ == '__main__':
    sim_num = 100

    a_list = [650]
    b_list = [575]
    it_nums = [10000]
    beta_0s = [.1, .5]
    beta_ns = [2, 1]
    n_0s = []
    save = True

    algos = ['metropolis']

    for algo in algos:
        for i in range(len(a_list)):
            a = a_list[i]
            b = b_list[i]
            N = a + b

            if check_a_b_relation(a, b):
                if algo == 'houdayer':
                    for n_0 in n_0s:
                        for it_num in it_nums:
                            s = time.time()
                            test(a, b, N, it_num, beta_0, beta_n, n_0, sim_num, algo, save=save)
                            print(f"Run Time: {round(time.time() - s, 5)}")
                else:
                    for i in range(len(beta_0s)):
                        for it_num in it_nums:
                            s = time.time()
                            test(a, b, N, it_num, beta_0s[i], 1, sim_num, algo, beta_n=beta_ns[i], save=save)
                            print(f"Run Time: {round(time.time() - s, 5)}")
