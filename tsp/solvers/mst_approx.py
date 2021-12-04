"""
Approximation solver using MST
"""
import sys, time

import numpy as np
import random
# import networkx
from heapq import heappush, heappop

class MSTApprox:
    def __init__(self, dist_mat, seed):
        self.dist_mat = np.array(dist_mat)
        self.seed = seed
        self.trace = []
        random.seed(self.seed)

    def prim_find_mst(self):
        N = len(self.dist_mat) # number of vertices
        cost = 0.0
        tree = set()
        start = random.randint(0, N-1)
        #print(start)
        queue = [(0, start, -1)] # priority queue for Prim's algorithm
        mst = {}

        while queue:
            weight, vtx, parent = heappop(queue)
            if vtx not in tree:
                tree.add(vtx)
                if parent >= 0:
                    if parent not in mst:
                        mst[parent] = []
                    if vtx not in mst:
                        mst[vtx] = []
                    mst[vtx].append(parent)
                    mst[parent].append(vtx)
                cost += weight
                for neigh in range(N):
                    if neigh not in tree:
                        heappush(queue, (self.dist_mat[vtx, neigh], neigh, vtx))
        self.mst_cost = cost
        self.mst = mst
    

    def solve(self):
        self.start_time = time.time()
        # First find an MST
        self.prim_find_mst()
        N = len(self.dist_mat)
        root = random.randint(0, N-1)

        # Add first node to path
        path = [root]
        seen = set()
        seen.add(root)
        pathcost = 0.0
        # Make a stack for traversing the MST
        stk = [root]
        while(stk):
            direct_edge = False
            for neigh in self.mst[stk[-1]]:
                if neigh not in seen:
                    seen.add(neigh)
                    pathcost += self.dist_mat[neigh, path[-1]]
                    path.append(neigh)
                    stk.append(neigh)
                    # We followed an edge in the MST
                    direct_edge = True
                    break
            if not direct_edge:
                # We have to skip vertices because we've seen all the neighbors
                stk.pop()
        pathcost += self.dist_mat[root, path[-1]]
        self.trace.append([time.time() - self.start_time, pathcost])

        self.best_dist = pathcost
        self.best_solution = np.array(path)

        # Just some bookkeeping to ensure 0 is the first vertex in path
        shift = np.where(self.best_solution == 0)[0][0]
        self.best_solution = np.roll(self.best_solution, -shift)
        
        return self.best_dist, self.best_solution, self.trace


if __name__ == '__main__':
    exit()
