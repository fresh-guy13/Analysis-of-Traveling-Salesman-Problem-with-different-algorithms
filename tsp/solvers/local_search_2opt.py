"""
Local search solver using 2-OPT exchange
"""

from itertools import combinations
import numpy as np
from numba import jit, njit
import random
import time


@njit
def feasible_edge(i, j, n):
    """
    Edges must not be sequential
    """
    if i > j:
        i, j = j, i
    if j >= n:
        return False
    if i == j or i + 1 == j or i == (j+1) % n:
        return False
    return True


@njit
def edge_pair(i, j):
    """
    Ensures i < j (used for flipping properly)
    """
    if i > j:
        i, j = j, i

    return i, j


@njit
def flip(tour, i, j):
    """
    Performs 2 opt flip
    
    Swaps edges (i,i+1) and (j,j+1) with edges (i,j) and (i+1,j+1).
    Note: an edge swap also reverses the paths between the swapped nodes
    """

    i, j = edge_pair(i, j)
    
    # Perform swap
    tour[i+1], tour[j] = tour[j], tour[i+1]

    # Reverse order between swap
    lo, hi = i+2, j-1
    while lo < hi:
        tour[lo], tour[hi] = tour[hi], tour[lo]
        lo += 1
        hi -= 1

    return tour


@njit
def flip_gain(adj_mat, tour, i, j):
    """
    Calculates benefit of removing edges (i,i+1) and (j,j+1)
    with (i,j) and (i+1,j+1).

    A positive gain is an improvement
    """

    n = len(adj_mat)
    i, j = edge_pair(i, j)

    weights_before = adj_mat[tour[i], tour[(i+1) % n]]
    weights_before += adj_mat[tour[j], tour[(j+1) % n]]
    weights_after = adj_mat[tour[i], tour[j]]
    weights_after += adj_mat[tour[(i+1) % n], tour[(j+1) % n]]

    # Ensure no underflow
    if weights_before > weights_after:
        return weights_before - weights_after
    else:
        return 0

    
def initialize_tour(adj_mat, greedy=True):
    """Initializes tour using greedy approach or randomization"""

    if greedy:
        tour = np.zeros(len(adj_mat), dtype=np.dtype('uint32'))
        unvisited = set(range(1, len(adj_mat)))
        for i in range(1, len(adj_mat)):
            best_weight = float('inf')
            # Find closest neighbor 
            for neighbor in unvisited:
                if adj_mat[i-1, neighbor] < best_weight:
                    tour[i] = neighbor
                    best_weight = adj_mat[i-1, neighbor]
            unvisited.remove(tour[i])
    else:
        # First node in tour is always starting city,
        # so only shuffle the last n-1 nodes
        tour = np.arange(len(adj_mat))
        np.random.shuffle(tour[1:])

    return tour


def tour_dist(adj_mat, tour):
    """
    Computes distance of tour
    """
    dist = 0
    for i in range(1, len(adj_mat)):
        dist += adj_mat[tour[i-1], tour[i]]
    dist += adj_mat[tour[-1], 0]
    return dist


def local_search_2opt(tsp_data, seed=None, max_time=float('inf'), niters=10, debug=False):
    """
    Local search 2-OPT solver
    """
    start_time = time.time()

    # Numpy and native python use different rng's
    np.random.seed(seed)
    random.seed(seed)
    
    adj_mat = tsp_data.to_adjacency_mat()
    n = len(adj_mat)
    
    def feasible_edges():
        """Returns random feasible edges"""
        return filter(
            lambda c: feasible_edge(c[0], c[1], n),
            combinations(random.sample(range(n), n), 2)
        )

    # Set for ensuring we don't converge to an already visited tour.
    # A tabu basically
    visited = set()

    best_tour = None
    best_dist = float('inf')
    trace = []

    while time.time() - start_time < max_time:
        
        # Initialize tour (random or greedy)
        greedy = False if random.random() < 0.5 else True
        tour = initialize_tour(adj_mat, greedy=True)
        if not trace:
            initial_dist = tour_dist(adj_mat, tour)
            trace.append([time.time() - start_time, initial_dist])
            if debug:
                print([initial_dist, trace[-1][0]])

        visited.add(str(tour))

        # Iterate until no improvement gained from 2-opt flipping
        improved = True
        while improved:
            improved = False
            
            best_improvement = 0
            best_flip = None

            # Whether to step to first improvement or best
            just_first_improvement = False if random.random() < 0.5 else True

            # Search for best flip
            for i, j in feasible_edges():
                gain = flip_gain(adj_mat, tour, i, j)
                if gain > best_improvement:
                    new_tour = flip(tour.copy(), i, j)
                    if str(new_tour) not in visited:
                        best_improvement = gain
                        best_flip = (i, j)
                        if just_first_improvement:
                            break

            # Update current tour
            if best_improvement > 0:
                i, j = best_flip
                flip(tour, i, j)
                visited.add(str(tour))
                improved = True

        # Determine if current tour is the best seen
        cur_dist = tour_dist(adj_mat, tour)

        if cur_dist < best_dist:
            trace.append([time.time() - start_time, cur_dist])
            if (debug):
                print([cur_dist, trace[-1][0]])
            best_tour = tour
            best_dist = cur_dist

    return best_dist, best_tour, trace


class LS2_2opt:
    """
    Solver interface
    """
    def __init__(self, tsp_data, seed=None, max_time=float('inf'), niters=10, debug=False):
        
        self.tsp_data = tsp_data
        self.seed = seed
        self.max_time = max_time
        self.niters = niters
        self.debug = debug

    def solve(self):
        return local_search_2opt(self.tsp_data, self.seed, self.max_time, self.niters, self.debug)
    


if __name__ == '__main__':

    from tsp.parse import parse
    import sys

    if (len(sys.argv) < 2):
        print("Usage: local_search_2opt.py <city>")
        exit(0)
        
    filename = f"../DATA/{sys.argv[1]}.tsp"
    
    d = parse(filename)
    sol = local_search_2opt(d)
    print(sol)
    
