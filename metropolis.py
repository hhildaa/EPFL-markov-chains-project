import numpy as np

from graph import assess_estimation_quality


def change_one_elem(state, ind):
    # change the color of the person's at the index
    changed = state.copy()
    changed[ind] = changed[ind] * (-1)
    return changed


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


def calculate_energy_change(state, chosen_person, adj, a, b, n):
    local_energy = 0
    edges = adj[chosen_person]
    for i in range(len(state)):
        if i != chosen_person:
            e = edges[i]
            h = calculate_h(a, b, n, e)
            local_energy += h * state[i]
    return 2 * state[chosen_person] * local_energy


def calculate_energy_change_lightning(state, chosen_person, adj, a, b, n):
    edges = adj[chosen_person]
    edges_inv = np.abs(edges - 1)
    local_energy = (1/2)*(edges * np.log(a / b) + edges_inv * np.log((1-a/n)/(1-b/n))) * state
    local_energy[chosen_person] = 0
    return 2 * state[chosen_person] * sum(local_energy)


def metropolis(start, adj, a, b, n, beta, x_star, it_num=100):
    state = start
    l = len(start)
    acceptance_rate_list = []
    estimation_overlaps = []
    for i in range(it_num):
        chosen_person = np.random.randint(l)

        change_energy = calculate_energy_change_lightning(state, chosen_person, adj, a, b, n)
        accept_rate = np.exp(-beta * change_energy)

        random = np.random.rand(1)

        if random <= accept_rate:
            new_state = change_one_elem(state, chosen_person)
            state = new_state
            acceptance_rate_list.append(1)
        else:
            acceptance_rate_list.append(0)

        # Current overlap
        estimation_overlaps.append(assess_estimation_quality(state, x_star))

    return acceptance_rate_list, estimation_overlaps


def houdayer(x_hat_1, x_hat_2, adj, a, b, n, beta, x_star, n_0=2, it_num=100):
    """
    If n_0 == 2, then we have the original houdayer algorithm, for n_0 > 2, it's mixed houdayer.
    """
    acceptance_rate_list = []
    estimation_overlaps_1 = []
    estimation_overlaps_2 = []

    for i in range(it_num):
        if i % n_0 == 0:
            x_hat_1, x_hat_2 = houdayer_step(x_hat_1, x_hat_2, adj)
        else:
            x_hat_1 = metropolis_step(x_hat_1, adj, a, b, n, beta)
            x_hat_2 = metropolis_step(x_hat_2, adj, a, b, n, beta)

        # Current overlap
        estimation_overlaps_1.append(assess_estimation_quality(x_hat_1, x_star))
        estimation_overlaps_2.append(assess_estimation_quality(x_hat_2, x_star))

    return estimation_overlaps_1, estimation_overlaps_2


def metropolis_step(state, adj, a, b, n, beta):
    # choose a random person
    l = len(state)
    chosen_person = np.random.randint(l)

    # get the new state by changing the color of the random person
    new_state = change_one_elem(state, chosen_person)

    # calculate the acceptance and go to the next iteration
    change_energy = calculate_energy_change_lightning(state, chosen_person, adj, a, b, n)
    accept_rate = np.exp(-beta * change_energy)

    random = np.random.rand(1)

    if random <= accept_rate:
        state = new_state
        # acceptance_rate_list.append(1)
    else:
        pass
        # acceptance_rate_list.append(0)

    return state


def houdayer_step(x_hat_1: np.array, x_hat_2: np.array, adj: np.array) -> np.array:
    y_hat = x_hat_1 * x_hat_2

    l = np.arange(0, len(y_hat), 1)
    ind_select = l[y_hat == -1]
    if len(ind_select) > 0:
        random = np.random.randint(len(ind_select))
        chosen = ind_select[random]

        neighbours = np.arange(0, adj.shape[0])[adj[chosen] == 1]
        mask = np.ones_like(x_hat_1)
        negative_neighs = neighbours[y_hat[neighbours] == -1]
        mask[negative_neighs] = mask[negative_neighs] * -1
        x_hat_1 = x_hat_1 * mask
        x_hat_2 = x_hat_2 * mask

    return x_hat_1, x_hat_2
