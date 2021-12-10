import numpy as np

# Baseline algorithm - simple random walk
from graph import assess_estimation_quality


def change_one_elem(state, ind):
    # change the color of the person's at the index
    changed = state.copy()
    changed[ind] = changed[ind] * (-1)
    return changed


def f_dummy(adj: np.array, state: np.array, ind: int):
    """
    this function counts the red and blue neighbours of the chosen person, and
    if the person has more of the person's colored neighbours, then it will
    suggest not to change color, otherwise yes
    """
    chosen = state[ind]

    # count the blue and the red neighbours
    neigh = adj[ind, :]

    # + ones
    red_num = neigh[state > 0].sum()
    # - ones
    blue_num = neigh[state < 0].sum()

    if chosen == 1:
        return (red_num) / (red_num + blue_num)
    else:
        return (blue_num) / (red_num + blue_num)


def calculate_h(a, b, n, e):
    t1 = e*np.log(a/b)
    t2 = (1-e)*np.log((1-a/n)/(1-b/n))
    return (1/2)*(t1 + t2)


def calculate_energy(x_hat, adj, a, b, n):
    energy = 0
    for i in range(len(adj)):
        for j in range(i+1, len(adj[i])):
            e = adj[i][j]
            x_i = x_hat[i]
            x_j = x_hat[j]
            h = calculate_h(a, b, n, e)
            energy += h * x_j * x_i

    return - energy


def baseline(start, adj, a, b, n, beta, x_star, it_num=100):
    state = start
    l = len(start)
    acceptance_rate_list = []
    epsilon = 0.0001
    estimation_overlaps = []
    for i in range(it_num):
        # choose a random person
        chosen_person = np.random.randint(l)

        # get the new state by changing the color of the random person
        new_state = change_one_elem(state, chosen_person)

        # calculate the acceptance and go to the next iteration
        change_energy = calculate_energy(new_state, adj, a, b, n) - calculate_energy(state, adj, a, b, n)
        accept_rate = np.exp(-beta * change_energy)

        random = np.random.rand(1)

        if random <= accept_rate:
            state = new_state
            acceptance_rate_list.append(1)
        else:
            acceptance_rate_list.append(0)

        # Current overlap
        estimation_overlaps.append(assess_estimation_quality(state, x_star))

    # visualize_graph(adj, state)
    return state, acceptance_rate_list, estimation_overlaps
