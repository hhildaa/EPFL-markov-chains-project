from graph import *
from metropolis import *
from plots import plot_acceptance_rates, visualize_graph, plot_estimation_overlaps_over_iterations


if __name__ == '__main__':
    a = 30
    b = 10
    N = a+b
    it_num = 100

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
        start = np.random.randint(2, size=N)
        start = np.array(np.where(start == 1, -1, 1))

        final_state, acceptance_rate_list, estimation_overlap = \
            baseline(start, adj_matrix, a, b, N, 1, x_star, it_num=it_num)

        acceptances_over_runs.append(acceptance_rate_list)
        estimation_overlap_over_runs.append(estimation_overlap)

        print(f'Iteration: {i + 1}/{sim_num} \t overlap: {round(estimation_overlap[-1] * 100, 2)}%')

    plot_estimation_overlaps_over_iterations(estimation_overlap_over_runs)
    # plot_acceptance_rates(acceptances_over_runs)
    # print(f"Average over all runs: {np.mean(average_over_runs)*100:.2f}% compared to random: {random_avg*100:.2f}%")
