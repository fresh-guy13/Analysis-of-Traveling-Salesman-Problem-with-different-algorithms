"""
Branch and Bound Solver

Note:
  This file includes a python bnb solver, but also adds an interface to a cpp
  implementation. The default is the cpp version.
"""

from collections import defaultdict
import copy
import heapq
import math
import numpy as np
from numba.experimental import jitclass
from numba import uint32, jit, njit
from numba.np.extensions import cross2d
import time
from heapq import heappush, heappop

from _solvers import branch_and_bound as bnb_cpp

@njit
def prim_find_mst(disjoint_items, dist_mat):
    """
    Calculates minimum spanning tree weight using Prim's algorithm
    """
    N = len(disjoint_items)
    cost = 0.0
    tree = Bitset(N)
    start = 0
    queue = [(np.uint32(0), start)]

    while queue:
        weight, vtx = heappop(queue)
        if not tree.contains(vtx):
            tree.add(vtx)
            cost += weight
            for j in range(N):
                neigh = disjoint_items[j]
                if not tree.contains(j):
                    heappush(queue, (dist_mat[disjoint_items[vtx], neigh], j))
    return cost


# For Bitset class below
spec = [
    ('_N', uint32),
    ('_bitsize', uint32),
    ('_bitset', uint32[:]),
    ('_len', uint32)
    ]

@jitclass(spec)
class Bitset:
    """
    Set datastructure with a small memory footprint to hold nonnegative integers 0...N-1
    
    Used for determining remaining nodes in the tour
    """
    def __init__(self, N: int):

        self._N = N
        self._bitsize = 32
        array_len = math.ceil(self._N / self._bitsize)
        self._bitset = np.zeros(array_len, dtype=np.dtype('uint32'))
        self._len = 0

    def len(self):
        return self._len
        
    def contains(self, item):
        if item < self._N:
            block_idx, elem_idx = divmod(item, self._bitsize)
            return (self._bitset[block_idx] & (1 << elem_idx)) > 0
        else:
            return False

    def add(self, item):
        """Add item to set"""
        if item < self._N:
            block_idx, elem_idx = divmod(item, self._bitsize)
            self._bitset[block_idx] |= (1 << elem_idx)
            self._len += 1

    def remove(self, item):
        """Remove item from set"""
        if item < self._N:
            block_idx, elem_idx = divmod(item, self._bitsize)
            self._bitset[block_idx] &= ~(1 << elem_idx)
            self._len -= 1

    def empty(self):
        for i in range(len(self._bitset)):
            if self._bitset[i] != 0:
                return False
        return True

    def npitems(self):
        arr = np.zeros(self._len, dtype=np.dtype('uint32'))
        j = 0
        for i in range(self._N):
            if self.contains(i):
                arr[j] = i
                j = j + 1
        return arr

    def items(self):
        return [i for i in range(self._N) if self.contains(i)]

    def copy(self):
        b = Bitset(self._N)
        b._bitset = self._bitset.copy()
        b._len = self._len
        return b
    

class Node:
    """
    Branch and bound node
    """
    
    def __init__(self, item, subproblem: Bitset, level=0, parent=None, distance=0):
        self.item = item
        self.subproblem = subproblem
        self.level = level
        self.parent = parent
        self.distance = distance
        self.lower_est = 0

        self.cost = float('inf')

    def __lt__(self, other):
        return self.priority < other.priority

    @property
    def priority(self):
        """Prioritize leaf nodes with a smaller lower bound"""
        return self.lowerbound / self.level
    
    @property
    def path(self):
        """
        Returns the path of the tour up until this node
        """
        if not hasattr(self, '_path'):
            path = []
            node = self
        
            while node is not None:
                path.append(node.item)
                node = node.parent
            
            self._path = list(reversed(path))
        return self._path

    def expand(self, adj_mat):
        """
        Iterator over each node in the current subproblem
        """
        level = self.level + 1
        for item in self.subproblem.items():
            new_subproblem = self.subproblem.copy()
            new_subproblem.remove(item)
            distance = self.distance + adj_mat[self.item, item]
            yield Node(
                item, new_subproblem, level=level, distance=distance, parent=self
            )

    @property
    def lowerbound(self):
        """
        Lower bound is current distance plus lower estimation on remaining nodes
        """
        return self.distance + self.lower_est

    def set_cost(self, cost):
        """
        Sets the total cost (should be done once a tour is complete).
        Cost is equal to the distance for a full tour, which includes traveling back to start.
        """
        self.cost = cost

    def set_lower_est(self, lower_est):
        """
        Set lower estimation on remaining nodes
        """
        self.lower_est = lower_est

@njit
def direction(a, b, c):
    """
    Find direction by computing cross product.
    Used for intersection code.
    """
    return cross2d(c-a, b-a)

@njit
def intersect(a, b, c, d):
    """
    Determine if line segments (a,b) and (c,d) intersect.
    Allow segments that share endpoints.
    """
    d1 = direction(c,d,a)
    d2 = direction(c,d,b)
    d3 = direction(a,b,c)
    d4 = direction(a,b,d)
    if (((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0))
        and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0))):
        return True
    return False
        
def branch_and_bound_py(tsp_data, max_time, debug=False, **kwargs):
    """
    Banch and bound solver
    """

    start_time = time.time()
    adj_mat = tsp_data.to_adjacency_mat()

    # Initialize unvisited vertices
    all_candidates = Bitset(len(adj_mat))
    for i in range(1, len(adj_mat)):
        all_candidates.add(i)

    # Level of terminal leaf nodes
    max_level = len(adj_mat) - 1

    # Priority queue holding most promising nodes
    F = [Node(0, all_candidates)]
    best_solution = F[0]

    idx = 0
    trace = []

    def no_intersections(node):
        """
        Helper code for determining if adding this node 
        results in an intersection among edges in tour
        """

        # If only 1 or 2 nodes in tour, then no intersection possible
        if node.parent is None or node.parent.parent is None:
            return True
        
        cur_edge_b = tsp_data.coords[node.item]
        cur_edge_a = tsp_data.coords[node.parent.item]

        # Loop through each previous edge in tour
        node = node.parent
        b = tsp_data.coords[node.item]
        while node.parent is not None:
            a = tsp_data.coords[node.parent.item]
            if intersect(a, b, cur_edge_a, cur_edge_b):
                return False

            node = node.parent
            b = a

        return True

    # Main branch-and-bound loop
    while F and (time.time()-start_time) < max_time:
        
        node = heapq.heappop(F)

        # Expand subproblem
        for subnode in node.expand(adj_mat):
            # Solution found
            if subnode.level == max_level:                
                subnode.set_cost(subnode.distance + adj_mat[subnode.item,0])
                # Best solution?
                if subnode.cost < best_solution.cost:
                    best_solution = subnode
                    trace.append([best_solution.cost, time.time()-start_time])
            else:
                # Optimal solution doesn't have intersecting edges
                if not no_intersections(subnode):
                    continue

                # Calculate lower estimation on remaining nodes
                lower_est = 0
                if not subnode.subproblem.empty():

                    disjoint_items = subnode.subproblem.npitems()
                    
                    # 1. Shortest edges from start and current node
                    from_start = np.min(adj_mat[0,disjoint_items])
                    from_curr = np.min(adj_mat[subnode.item, disjoint_items])

                    lower_est += from_start + from_curr

                    # 2. Minimum spanning tree of remaining elements
                    cost = prim_find_mst(disjoint_items, adj_mat)
                    lower_est += int(cost)
                    
                subnode.set_lower_est(lower_est)

                # Not a dead end, add node to queue
                if subnode.lowerbound < best_solution.cost:
                    if debug:
                        print(subnode.level, subnode.lowerbound, best_solution.cost, idx, int(time.time()-start_time))
                    heapq.heappush(F, subnode)
                    idx += 1

    return best_solution.cost, best_solution.path, trace


def branch_and_bound_cpp(tsp_data, max_time, depth_first=False, debug=False):
    """
    Interface to cpp branch-and-bound implementation
    """
    solution = bnb_cpp(tsp_data.coords, max_time, depth_first, debug)
    best_tour = solution.tour
    best_dist = solution.distance

    trace = []
    for trace_item in solution.trace:
        trace.append([trace_item.distance, trace_item.time])

    return best_dist, best_tour, trace


def branch_and_bound(tsp_data, max_time, debug=False, lang='cpp'):
    """
    Forwards args to selected language's implementation
    """
    if lang == 'cpp':
        return branch_and_bound_cpp(tsp_data, max_time, debug=debug)
    elif lang == 'py':
        return branch_and_bound_py(tsp_data, max_time, debug=debug)

    
class BranchAndBound:
    """
    Solver interface
    """
    def __init__(self, tsp_data, max_time, debug=False, lang='cpp'):
        self.tsp_data = tsp_data
        self.max_time = max_time
        self.debug = debug
        self.lang = lang

    def solve(self):
        return branch_and_bound(self.tsp_data, self.max_time, self.debug, self.lang)
    

if __name__ == '__main__':

    import sys
    import time
    
    from tsp.parse import parse

    if (len(sys.argv) < 2):
        print("Usage: branch_and_bound.py <city>")
        exit(0)
    
    filename = f"../DATA/{sys.argv[1]}.tsp"
    d = parse(filename)
    
    sol = branch_and_bound(d, 600)
    print(sol)
    
    
    
