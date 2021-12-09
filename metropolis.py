import numpy as np

# Baseline algorithm - simple random walk


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

    # if state[ind]==1:  # the current state is red
    # majority of neighbours are red
    #  if red_num > blue_num:
    #    return red_num
    # majority of neighbours are blue
    #  else:
    #    return -blue_num
    # else:
    # majority of neighbours are red
    #  if red_num > blue_num:
    #    return blue_num
    # majority of neighbours are blue
    #  else:
    #    return -red_num


def baseline(start, adj, beta, it_num=100):
    # initialize start state
    state = start
    l = len(start)
    acceptance_rate_list = []
    for i in range(it_num):
        # choose a random person
        chosen_person = np.random.randint(l)

        # get the new state by changing the color of the random person
        new_state = change_one_elem(state, chosen_person)

        # calculate the acceptance and go to the next iteration
        # accept_rate = np.exp(-beta * (f_dummy(adj, new_state, chosen_person) - f_dummy(adj, state, chosen_person)))

        x = f_dummy(adj, new_state, chosen_person)
        y = f_dummy(adj, state, chosen_person)
        accept_rate = f_dummy(adj, new_state, chosen_person) / f_dummy(adj, state, chosen_person)
        random = np.random.rand(1)

        if random <= accept_rate:
            state = new_state
            acceptance_rate_list.append(1)
        else:
            acceptance_rate_list.append(0)

    # visualize_graph(adj, state)
    return state, acceptance_rate_list
