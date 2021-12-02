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
        N = len(self.dist_mat)
        cost = 0.0
        tree = set()
        start = random.randint(0, N-1)
        #print(start)
        queue = [(0, start, -1)]
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
        self.prim_find_mst()
        N = len(self.dist_mat)
        root = random.randint(0, N-1)
        path = [root]
        seen = set()
        seen.add(root)
        pathcost = 0.0
        stk = [root]
        while(stk):
            direct_edge = False
            for neigh in self.mst[stk[-1]]:
                if neigh not in seen:
                    seen.add(neigh)
                    pathcost += self.dist_mat[neigh, path[-1]]
                    path.append(neigh)
                    stk.append(neigh)
                    direct_edge = True
                    break
            if not direct_edge:
                stk.pop()
        pathcost += self.dist_mat[root, path[-1]]
        self.trace.append([time.time() - self.start_time, pathcost])

        self.best_dist = pathcost
        self.best_solution = np.array(path)

        # Just some bookkeeping to ensure 0 is the first vertex in path
        shift = np.where(self.best_solution == 0)[0][0]
        self.best_solution = np.roll(self.best_solution, -shift)
        
        return self.best_dist, self.best_solution, self.trace


        # odd_degree_vertices = sorted([vtx for vtx, neighbors in self.mst.items() if len(neighbors) % 2 != 0])
        # sub_dist_mat = self.dist_mat[np.ix_(odd_degree_vertices, odd_degree_vertices)]
        # sub_dist_mat = np.array(sub_dist_mat, dtype=np.float64)
        # sub_dist_mat[sub_dist_mat==0.] = np.inf
        # recip_weights = np.reciprocal(sub_dist_mat)
        
        # print(self.mst)
        # print (self.mst_cost)
        # G = networkx.convert_matrix.from_numpy_array(self.dist_mat)
        # T = networkx.minimum_spanning_tree(G, weight='weight', algorithm='kruskal', ignore_nan=False)
        # print(T.size(weight='weight'))
        # print(sorted(T.edges(data=True)))

if __name__ == '__main__':
    exit()
