"""
Local search with simulated annealing

TODO: Description
"""

import sys, time
from . import utils
import numpy as np
import math
import random
import itertools

class LS1_SA(object):
    def __init__(self, dist_mat, cooling_rate, seed, max_time):
        self.dist_mat = dist_mat
        self.candidate_array = np.asarray([i for i in range(1,len(self.dist_mat) + 1)])
        self.cooling_rate = cooling_rate
        self.best_solution = np.copy(self.candidate_array)
        self.best_dist = utils.cal_total_dist(self.best_solution, self.dist_mat)
        #When temperature is 10000, performs bad
        self.Temp = 10000
        self.seed = seed
        self.max_time = max_time
        self.trace = []
        
    def RestartToBest(self):
        return 0, np.copy(self.best_solution), np.copy(self.best_dist)

    def SA_process(self):
        n_iters = 0
        j = 0
        NRAND = 10000
        N = len(self.candidate_array)

        # Generate all pairs of distinct indices with no repeats
        all_pairs = np.array(list(itertools.combinations(range(N), 2)))
        # Get a supply of random numbers which are used to index into the array of all pairs
        random_supply = self.rng.integers(low=0, high=all_pairs.shape[0], size=NRAND)

        curr_solution = utils.gen_random_ans(self.candidate_array)
        curr_dist = utils.cal_total_dist(curr_solution, self.dist_mat)
        exe_time = 0
        #Count the number of staying in the same state
        stay_num = 0
        start_time = time.time()
        while self.Temp >= 1 and self.max_time - exe_time > 0:
            n_iters = n_iters + 1

            # get a random pair
            r_pair = all_pairs[random_supply[j],:]
            # swap
            curr_solution[r_pair[0]], curr_solution[r_pair[1]] = curr_solution[r_pair[1]], curr_solution[r_pair[0]]
            
            next_dist = utils.cal_total_dist(curr_solution, self.dist_mat)
            if next_dist < curr_dist :
                curr_dist = next_dist
                stay_num = 0
                if curr_dist < self.best_dist:
                    self.best_solution, self.best_dist = np.copy(curr_solution), curr_dist
                    #Calculate trace_time
                    trace_time = time.time() - start_time
                    self.trace.append([trace_time, int(self.best_dist)])
            else:
                if np.random.uniform(0, 1) < math.exp((self.best_dist - curr_dist)/self.Temp) :
                    # Accept candidate solution: retain swap
                    curr_dist = next_dist
                else:
                    # Reject candidate solution: undo swap
                    curr_solution[r_pair[0]], curr_solution[r_pair[1]] = curr_solution[r_pair[1]], curr_solution[r_pair[0]]
                stay_num += 1
            
            j += 1
            # Replenish supply of random numbers
            if j >= NRAND:
                j = 0
                random_supply = self.rng.integers(low=0, high=all_pairs.shape[0], size=NRAND)

            
            #If we can't get the better solution for 1000 time, go to best solution
            if stay_num >= 1000:
                stay_num, curr_solution, curr_dist = self.RestartToBest()
            self.Temp *= (1 - self.cooling_rate)
            exe_time = time.time() - start_time
            
        #print(n_iters)
        
    def Simulated_Annealing(self):
        #Set seed for the randomness
        np.random.seed(self.seed)
        self.rng = np.random.default_rng(self.seed)

        #Generate random solution for the problem
        self.SA_process()

        # Just some bookkeeping to ensure 0 is the first vertex in path
        self.best_solution -= np.min(self.best_solution) # ensure 0-index
        shift = np.where(self.best_solution == 0)[0][0]
        self.best_solution = np.roll(self.best_solution, -shift)
        
        return self.best_dist, self.best_solution, self.trace

    def solve(self):
        return self.Simulated_Annealing()


if __name__ == '__main__':
    # Temporarily use Atlanta as the example

    # TSP_Data = parse("DATA/Atlanta.tsp")
    # test = LS1_SA(TSP_Data.to_adjacency_mat(), 0.001, np.random.randint(1, 10000))
    # solution = test.Simulated_Annealing()
    print("Using Simulated Annealing to solve TSP\n")
    

