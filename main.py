from graph import *
from metropolis import *
from plots import plot_acceptance_rates, visualize_graph, plot_estimation_overlaps_over_iterations


if __name__ == '__main__':
    a = 30
    b = 10
    N = a+b
    it_num = 100
    beta = 1

    check_a_b_relation(a, b)

    x_star = generate_x(N, 5)
    adj_matrix = generate_adjacency_matrix(x_star, a, b, N)
    random_avg = check_random_estimates(a, b, N, x_star, verbose=False)

    sim_num = 10
    acceptances_over_runs = []
    estimation_overlap_over_runs = []

    print(f"=== a: {a}  b: {b}  N: {N} iterations per run: {it_num} ===")

    for i in range(sim_num):
        # starting position: random assignment of clusters
        x_hat_1 = generate_x_hat(N)
        x_hat_2 = generate_x_hat(N)

        final_state = \
            houdayer(x_hat_1, x_hat_2, adj_matrix, a, b, N, beta, x_star, it_num=it_num)

        #acceptances_over_runs.append(acceptance_rate_list)
        #estimation_overlap_over_runs.append(estimation_overlap)
        #estimation_overlap[-1]

        print(f'Iteration: {i + 1}/{sim_num} \t overlap: {round( 100, 2)}%')

    plot_estimation_overlaps_over_iterations(estimation_overlap_over_runs)
    # plot_acceptance_rates(acceptances_over_runs)
    # print(f"Average over all runs: {np.mean(average_over_runs)*100:.2f}% compared to random: {random_avg*100:.2f}%")
