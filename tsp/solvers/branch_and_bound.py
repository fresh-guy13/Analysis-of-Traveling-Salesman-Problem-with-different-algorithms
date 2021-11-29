"""
Branch and Bound Solver
"""

from collections import defaultdict
import copy
import heapq
import math
import numpy as np
from numba.experimental import jitclass
from numba import uint32, jit, njit
from numba.np.extensions import cross2d
from scipy.sparse.csgraph import minimum_spanning_tree
import time

spec = [
    ('_N', uint32),
    ('_bitsize', uint32),
    ('_bitset', uint32[:]),
    ('_len', uint32)
    ]

@jitclass(spec)
class Bitset:
    
    def __init__(self, N: int):

        self._N = N
        self._bitsize = 32
        array_len = math.ceil(self._N / self._bitsize)
        self._bitset = np.zeros(array_len, dtype=np.dtype('uint32'))
        self._len = 0

    def len(self):
        return self._len
        
    def contains(self, item):
        """Magic function used by 'in' syntax"""
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

    def items(self):
        return [i for i in range(self._N) if self.contains(i)]

    def copy(self):
        b = Bitset(self._N)
        b._bitset = self._bitset.copy()
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
        return self.lowerbound / self.level
    
    @property
    def path(self):
        if not hasattr(self, '_path'):
            path = []
            node = self
        
            while node is not None:
                path.append(node.item)
                node = node.parent
            
            self._path = list(reversed(path))
        return self._path

    def expand(self, adj_mat):
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
        # TODO update this
        return self.distance + self.lower_est

    def set_cost(self, cost):
        self.cost = cost

    def set_lower_est(self, lower_est):
        self.lower_est = lower_est

@njit
def direction(a, b, c):
    return cross2d(c-a, b-a)

@njit
def intersect(a, b, c, d):
    """
    Allow segments that share endpoints
    """
    # if a == c or a == d or b == c or b == d:
    #     return False
    d1 = direction(c,d,a)
    d2 = direction(c,d,b)
    d3 = direction(a,b,c)
    d4 = direction(a,b,d)
    if (((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0))
        and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0))):
        return True
    return False
    
        
def branch_and_bound(tsp_data):
    """
    TODO implement
    """

    start_time = time.time()
    
    adj_mat = tsp_data.to_adjacency_mat()
    
    all_candidates = Bitset(len(adj_mat))
    for i in range(1, len(adj_mat)):
        all_candidates.add(i)

    max_level = len(adj_mat) - 1
        
    F = [Node(0, all_candidates)]
    best_solution = F[0]

    first_sol_found = True

    idx = 0

    def no_intersections(node):

        if node.parent is None or node.parent.parent is None:
            return True

        cur_edge_b = tsp_data.coords[node.item]
        cur_edge_a = tsp_data.coords[node.parent.item]

        # other = LineString([cur_edge_a, cur_edge_b])
        # line = LineString([tsp_data.coords[i] for i in node.parent.path])
        # return line.intersects(other)

        node = node.parent
        b = tsp_data.coords[node.item]
        while node.parent is not None:
            a = tsp_data.coords[node.parent.item]
            if intersect(a, b, cur_edge_a, cur_edge_b):
                return False

            node = node.parent
            b = a

        return True
    
    while F:
                
        # Use depth-first-search until a first solution is found
        if not first_sol_found:
            node = F.pop()
        # Then use best-first search
        else:
            node = heapq.heappop(F)

        #print(node.level)
        
        for subnode in node.expand(adj_mat):
            if subnode.level == max_level:
                print("Solution found")
                
                subnode.set_cost(subnode.distance + adj_mat[subnode.item,0])
                if subnode.cost < best_solution.cost:
                    best_solution = subnode
                    #print(best_solution.path)
                    if not first_sol_found:
                        first_sol_found = True
                        heapq.heapify(F)
                        print("Found first solution")

                #print(subnode.level, subnode.lowerbound, best_solution.cost, idx)
            else:
                # Optimal solution doesn't have intersecting edges
                if not no_intersections(subnode):
                    continue

                # Calculate lower estimation on remaining nodes
                lower_est = 0
                if not subnode.subproblem.empty():

                    disjoint_items = subnode.subproblem.items()
                    
                    # 1. Shortest edges from start and current node
                    from_start = float('inf')
                    from_curr = float('inf')
                    for i in disjoint_items:
                        if adj_mat[0, i] < from_start:
                            from_start = adj_mat[0, i]
                        if adj_mat[subnode.item, i] < from_curr:
                            from_curr = adj_mat[subnode.item, i]
                    lower_est += from_start + from_curr

                    # 2. Minimum spanning tree of remaining elements
                    nleft = len(disjoint_items)
                    tmp_adj = np.zeros((nleft, nleft), dtype=np.dtype('uint32'))
                    for i in range(nleft):
                        for j in range(nleft):
                            if i != j:
                                ii, jj = disjoint_items[i], disjoint_items[j]
                                tmp_adj[i,j] = adj_mat[ii,jj]

                    tmp_adj = minimum_spanning_tree(tmp_adj, overwrite=True)
                    mst_cost = int(tmp_adj.sum())
                                                        
                    lower_est += mst_cost
                    
                subnode.set_lower_est(lower_est)
                
                if subnode.lowerbound < best_solution.cost:
                    
                    print(subnode.level, subnode.lowerbound, best_solution.cost, idx, int(time.time()-start_time))
                    if not first_sol_found:
                        F.append(subnode)
                    else:
                        heapq.heappush(F, subnode)
                    idx += 1

    return best_solution


if __name__ == '__main__':

    import sys
    import time
    
    sys.path.append("..")
    from tsp import parse
    
    #filename = "../DATA/Roanoke.tsp"
    filename = "../DATA/Atlanta.tsp"

    d = parse(filename)

    sol = branch_and_bound(d)

    print(sol.cost)
    
    
