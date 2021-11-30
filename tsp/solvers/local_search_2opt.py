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
    
    if i > j:
        i, j = j, i

    return i, j
    
@njit
def flip(tour, i, j):
    """
    Performs 2 opt flip
    
    Swaps edges (i,i+1) and (j,j+1) with edges (i,j) and (i+1,j+1)
    Assume i < j
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
    dist = 0
    for i in range(1, len(adj_mat)):
        dist += adj_mat[tour[i-1], tour[i]]
    dist += adj_mat[tour[-1], 0]
    return dist

def local_search_2opt(tsp_data, seed=None, max_time=float('inf'), p=1e-5, niters=10):
    """
    p: probability of accepting worse solution
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
        
    visited = set()

    best_tour = None
    best_dist = float('inf')
    trace = []
    
    for _ in range(niters):
        
        tour = initialize_tour(adj_mat)
        visited.add(str(tour))
        
        improved = True
        while improved:
            improved = False
            
            best_improvement = 0
            best_flip = None

            for i, j in feasible_edges():
                gain = flip_gain(adj_mat, tour, i, j)
                if gain > best_improvement:
                    new_tour = flip(tour.copy(), i, j)
                    if str(new_tour) not in visited:
                        best_improvement = gain
                        best_flip = (i, j)

            if best_improvement > 0:
                i, j = best_flip
                flip(tour, i, j)
                visited.add(str(tour))
                improved = True

        cur_dist = tour_dist(adj_mat, tour)
        if cur_dist < best_dist:
            trace.append([cur_dist, time.time() - start_time])
            print(trace[-1])
            best_tour = tour
            best_dist = cur_dist

    return best_tour, best_dist, trace


if __name__ == '__main__':

    from parse import parse
    import sys

    filename = f"../DATA/{sys.argv[1]}.tsp"
    
    #filename = "../DATA/Roanoke.tsp"
    #filename = "../DATA/Atlanta.tsp"
    #filename = "../DATA/Denver.tsp"
    
    d = parse(filename)
    
    s = time.time()

    sol = local_search_2opt(d)
    
