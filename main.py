from graph import *
from metropolis import *
from plots import plot_acceptance_rates, visualize_graph, plot_estimation_overlaps_over_iterations
import pickle

DATA_PATH = './data/'



def test(a, b, N, it_num, beta, n_0, sim_num, algo): 
    x_star = generate_x(N, 5)
    adj_matrix = generate_adjacency_matrix(x_star, a, b, N)
    random_avg = check_random_estimates(a, b, N, x_star, verbose=False)

    estimation_overlap_over_runs_1 = []
    estimation_overlap_over_runs_2 = []

    acceptance_rate_list = None

    print(f"=== a: {a}  b: {b}  N: {N} iterations per run: {it_num} ===")

    if algo == 'houdayer': 
        for i in range(sim_num):
        # starting position: random assignment of clusters
            x_hat_1 = generate_x_hat(N)
            x_hat_2 = generate_x_hat(N)

            estimation_overlaps_1, estimation_overlaps_2 = \
                houdayer(x_hat_1, x_hat_2, adj_matrix, a, b, N, beta, x_star, n_0=n_0, it_num=it_num)

            estimation_overlap_over_runs_1.append(estimation_overlaps_1)
            estimation_overlap_over_runs_2.append(estimation_overlaps_2)

            print(f'Iteration: {i + 1}/{sim_num} \t overlap: {round(estimation_overlaps_1[-1] * 100, 2)}')

    else: 
        for i in range(sim_num):
        # starting position: random assignment of clusters
            x_hat_1 = generate_x_hat(N)

            _, estimation_overlaps_1 = \
                metropolis(x_hat_1, adj_matrix, a, b, N, beta, x_star, it_num=it_num)

            estimation_overlap_over_runs_1.append(estimation_overlaps_1)

            print(f'Iteration: {i + 1}/{sim_num} \t overlap: {round(estimation_overlaps_1[-1] * 100, 2)}')

    save_dict = {'a': a, 'b':b, 'N':N, 'it_num': it_num, 'beta': beta, 'n_0': n_0, 'sim_num': sim_num, 'algo': algo, 'estimation_overlap_over_runs_1': estimation_overlap_over_runs_1, 'estimation_overlap_over_runs_2': estimation_overlap_over_runs_2}

    with open( DATA_PATH + f'a_{a}_b_{b}_algo_{algo}_beta_{beta}_n0_{n_0}'+'.pickle', 'wb' ) as handle:
        pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == '__main__':
    a = 100
    b = 20
    N = a+b
    it_num = 1000
    beta = 1
    n_0 = 20
    algo = 'metropolis'
    sim_num = 1

    check_a_b_relation(a, b)

    test(a, b, N, it_num, beta, n_0, sim_num, algo)


