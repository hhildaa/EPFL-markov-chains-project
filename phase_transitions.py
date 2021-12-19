import numpy as np

from graph import generate_x_hat, generate_x, generate_adjacency_matrix
from metropolis import metropolis, houdayer
from phase_plots import theoretical_phase_transition
from utils import save_pickle, DATAPATH

DATA_PATH = './data/'

if __name__ == '__main__':
    a_list = None
    b_list = None
    algorithm = 'metropolis'
    rounds = 10
    n_0 = 10

    for N in [500]:  # 100, 500, 1000

        a_list = np.array([9.5, 8.5, 8, 7.5, 7, 5])
        b_list = np.array([0.5, 1.5, 2, 2.5, 3, 5])
        node_degree = int((b_list[0] + a_list[0])/2)

        it_nums = np.array([670000, 670000, 670000, 670000, 670000, 10], dtype=int)
        it_nums = it_nums.tolist()
        assert len(a_list) == len(b_list) == len(it_nums)

        beta = 1
        avg_estimation_overlaps = []

        r_crit = theoretical_phase_transition(a_list[0], b_list[0])
        print(f'Critical phase transition expected at r_crit = {round(r_crit, 4)}.')

        for idx in range(0, len(a_list)):  # len(a_list)):
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
                    _, estimation_overlaps = \
                        metropolis(x_hat_1, adj_matrix, a, b, N, beta, x_star, it_num=it_num)
                    estimation_overlap_over_runs.append(estimation_overlaps[-1])

                elif algorithm == "houdayer":
                    x_hat_2 = generate_x_hat(N)
                    estimation_overlaps, estimation_overlaps_2 = \
                        houdayer(x_hat_1, x_hat_2, adj_matrix, a, b, N, beta, x_star, n_0=n_0, it_num=it_num)
                    estimation_overlap_over_runs.append(estimation_overlaps[-1])
                    estimation_overlap_over_runs_2.append(estimation_overlaps_2[-1])

                print(f'{algorithm} iteration: {experiment_run + 1}/ {rounds} \t overlap: {round(estimation_overlaps[-1], 4)}')

            print(round(sum(estimation_overlap_over_runs) / len(estimation_overlap_over_runs), 4))
            if algorithm == "metropolis":
                avg_estimation_overlaps.append(estimation_overlap_over_runs)
            elif algorithm == "houdayer":
                chain_1 = np.array(estimation_overlap_over_runs)
                chain_2 = np.array(estimation_overlap_over_runs_2)
                avg_estimation_overlaps.append(np.mean([chain_1, chain_2], axis=0))

        b_div_a = np.array(b_list) / np.array(a_list)
        save_dict = {'b_div_a': b_div_a,
                     'N': N,
                     'algo': algorithm,
                     'avg_estimation_overlap': avg_estimation_overlaps,
                     'degree': node_degree}

        if algorithm == "houdayer":
            save_pickle(save_dict, DATAPATH, f'algo_{algorithm}_N_{N}_n0_{n_0}_degree_{node_degree}' + '.pickle')
        elif algorithm == "metropolis":
            save_pickle(save_dict, DATAPATH, f'algo_{algorithm}_N_{N}_degree_{node_degree}_FINAL' + '.pickle')
            # save_pickle(save_dict, DATAPATH, f'algo_{algorithm}_N_{N}_degree_{node_degree}_TEST' + '.pickle')
