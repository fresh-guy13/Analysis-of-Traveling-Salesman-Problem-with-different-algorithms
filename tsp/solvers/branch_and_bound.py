"""
Branch and Bound Solver
"""

from graph import TspData
import heapq
import numpy as np

class Node:
    """
    Branch and bound node

    TODO come up with a more efficient datastructure
    """
    def __init__(self, cost=float('inf'), path=[]):
        self.path = []
        self.cost = cost

    def __lt__(self, other):
        return self.cost < other.cost
        

def branch_and_bound(mat, func):
    """
    TODO implement
    """
    all_candidates = set(range(1, len(mat)))
    path = [0]

    F = [Node(cost=0, path=path)]
    best_solution = Node()

    while F:
        node = heapq.heappop(F)

        if len(node.path) == len(mat):
            # Only one way to go from here, back home
            node.cost += mat[node.path[-1],0]
            node.path.append(0)
            best_solution = min(best_solution, node)

        else:
            candidates = all_candidates - set(node.path)
            for candidate in candidates:
                new_path = node.path + [candidate]
                new_cost = node.cost + mat[node.path[-1], candidate]
                heapq.heappush(F, Node(new_path, new_cost))
            

    

    
