import numpy as np
from numpy.random import default_rng


def check_a_b_relation(a: int, b: int) -> None:
    if (a - b) ** 2 <= 2 * (a + b):
        print("We cannot detect the communitites in the limit." +
              " Chose a higher a or a lower b.")
    else:
        print("You chose a and b wisely.")


def generate_x(n: int, seed: int) -> np.array:
    rng = default_rng(seed=seed)
    x = rng.uniform(0, 1, n)
    x = [1 if x < 0.5 else -1 for x in x]
    return np.array(x)


def generate_adjacency_matrix(x_star: np.array, a: int, b: int, n: int) -> np.array:
    """
    Generate the connections to other nodes of each individual nodes, based on whether the other node
    is in the same cluster or not. Our approach works with masking.
    :param x_star: the cluster membership of each node
    :type x_star: np.array
    :param a: a/n: probability, that members of the same cluster share an edge
    :type a: int
    :param b: b/n: probability, that members of different clusters share an edge
    :type b: int
    :param n: number of total nodes
    :type n: int
    :return: the adjacency matrix
    :rtype: np.array
    """
    shape = len(x_star)
    x_star_matrix = np.outer(x_star, x_star)
    # Extract x_star * x_star = +1: these users are from the same group
    a_m = (x_star_matrix + 1) / 2

    # Extract x_star * x_star = -1: these users are from different group. Make it positive for edges.
    b_m = (x_star_matrix - 1) / 2 * (-1)

    # Generate P(e_ij = 1) given a and b
    # We generate too many and will later just keep the ones at the right spot
    rng = default_rng(seed=7)
    # Matrix of connected users given that they are in the same group
    a_e_ij = rng.choice(a=[0, 1], size=(shape, shape), p=[1 - a / n, a / n])
    # Matrix of connected users given that they are in different groups
    b_e_ij = rng.choice(a=[0, 1], size=(shape, shape), p=[1 - b / n, b / n])

    # We have to make sure the matrices of connected users are symmetric:
    # Only consider one lower triangle of the generated a-edge and b-edge matrix
    tri_matrix = np.tri(shape, k=-1)

    # Mirror the generated edge matrices
    a_e_ij = a_e_ij * tri_matrix + (a_e_ij * tri_matrix).T
    b_e_ij = b_e_ij * tri_matrix + (b_e_ij * tri_matrix).T
    # Merge the edge matrices based on there was an edge or not
    # Basically use the generated a_m/ b_m and a_e_ij/ b_e_ij as a mask
    # A node is always connected to itself
    edges = a_e_ij * a_m + b_e_ij * b_m + np.eye(shape)
    return edges


def check_edges(x_star: np.array, adj_list: np.array) -> None:
    n = len(x_star)
    x_star_matrix = np.outer(x_star, x_star)

    a_m = (x_star_matrix + 1) / 2
    b_m = np.abs((x_star_matrix - 1) / 2)

    # total number of a to a edges and b to b edges
    a_to_a_num = np.sum(a_m)
    b_to_b_num = np.sum(b_m)

    # edges from a to a and b to b that were actually connected by the adj list
    a_to_a_edge = np.sum(a_m * adj_list)
    b_to_b_edge = np.sum(b_m * adj_list)

    # this average should be ~a/N and ~b/N
    a_prob = a_to_a_edge / a_to_a_num
    b_prob = b_to_b_edge / b_to_b_num

    print(
        f"a/N prob: {a_prob * 100:.1f}% and a ~ {int(n * a_prob)} \nb/N prob: {b_prob * 100:.1f}% and b ~ {int(n * b_prob)}")


def assess_estimation_quality(n: int, x_hat: np.array, x_star: np.array) -> int:
    q_n = 1 / n * np.abs(np.sum(x_hat * x_star))
    return q_n


def check_random_estimates(a: int, b: int, n: int, x_star: np.array, verbose=False):
    random_estimation = []
    for i in range(10, 110):
        x_hat = generate_x(n, i)
        random_estimation.append(assess_estimation_quality(n, x_hat, x_star))
    if verbose:
        print(f"For a = {a}, b = {b}, N = {n} and random estimation "
              f"we get a score: {np.mean(random_estimation) * 100:.2f}%")
    return np.mean(random_estimation)
