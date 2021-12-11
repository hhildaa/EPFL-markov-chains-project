from graph import *
from metropolis import *
from plots import plot_acceptance_rates, visualize_graph, plot_estimation_overlaps_over_iterations


if __name__ == '__main__':
    a = 100
    b = 20
    N = a+b
    it_num = 1000
    beta = 1
    n_0 = 20

    check_a_b_relation(a, b)

    x_star = generate_x(N, 5)
    adj_matrix = generate_adjacency_matrix(x_star, a, b, N)
    random_avg = check_random_estimates(a, b, N, x_star, verbose=False)

    sim_num = 1
    estimation_overlap_over_runs_1 = []
    estimation_overlap_over_runs_2 = []

    acceptance_rate_list = None

    print(f"=== a: {a}  b: {b}  N: {N} iterations per run: {it_num} ===")

    for i in range(sim_num):
        # starting position: random assignment of clusters
        x_hat_1 = generate_x_hat(N)
        x_hat_2 = generate_x_hat(N)

        estimation_overlaps_1, estimation_overlaps_2 = \
            houdayer(x_hat_1, x_hat_2, adj_matrix, a, b, N, beta, x_star, n_0=n_0, it_num=it_num)

        estimation_overlap_over_runs_1.append(estimation_overlaps_1)
        estimation_overlap_over_runs_2.append(estimation_overlaps_2)

        print(f'Iteration: {i + 1}/{sim_num} \t overlap: {round(estimation_overlaps_1[-1] * 100, 2)}')

    # Plot convergences of our chains
    plot_estimation_overlaps_over_iterations(estimation_overlap_over_runs_1)
    plot_estimation_overlaps_over_iterations(estimation_overlap_over_runs_2)

    # Only plot the last chain's acceptance rates
    # plot_acceptance_rates(acceptance_rate_list)
