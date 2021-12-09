from graph import *
from metropolis import *
from plots import plot_acceptance_rates

if __name__ == '__main__':
    a = 15
    b = 1
    N = a+b
    it_num = 10000

    check_a_b_relation(a, b)

    x_star = generate_x(N, 5)
    adj_matrix = generate_adjacency_matrix(x_star, a, b, N)
    random_avg = check_random_estimates(a, b, N, x_star, verbose=True)

    sim_num = 10
    average_over_runs = []
    acceptances_over_runs = []
    print(f"=== a: {a}  b: {b}  N: {N} iterations per run: {it_num} ===")
    for i in range(sim_num):
        # starting position: random assignment of clusters
        start = np.random.randint(2, size=N)
        start = np.array(np.where(start == 1, -1, 1))

        final_state, acceptance_rate_list = baseline(start, adj_matrix, 1, it_num=it_num)
        overlap = assess_estimation_quality(N, final_state, x_star)
        average_over_runs.append(overlap)
        acceptances_over_runs.append(acceptance_rate_list)
        print(f'Iteration: {i + 1}/{sim_num} \t overlap: {round(overlap * 100, 2)}%')

    plot_acceptance_rates(acceptances_over_runs[0])
    print(f"Average over all runs: {np.mean(average_over_runs)*100:.2f}% compared to random: {random_avg*100:.2f}%")
