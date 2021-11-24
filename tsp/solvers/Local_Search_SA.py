import sys, time
from . import utils
sys.path.append("./tsp/")
from parse import parse

import numpy as np
import math
class LS1_SA(object):
    def __init__(self, dist_mat, cooling_rate, seed, max_time):
        self.dist_mat = dist_mat
        self.candidate_array = np.asarray([i for i in range(1,len(self.dist_mat) + 1)])
        self.cooling_rate = cooling_rate
        self.best_solution = np.copy(self.candidate_array)
        self.best_dist = utils.cal_total_dist(self.best_solution, self.dist_mat)
        self.Temp = 10000
        self.seed = seed
        self.max_time = max_time
        self.trace = []
    def RestartToBest(self):
        return 0, self.best_solution, self.best_dist

    def SA_process(self, Temp):
        start_time = time.time()
        curr_solution = utils.gen_random_ans(self.candidate_array)
        curr_dist = utils.cal_total_dist(curr_solution, self.dist_mat)
        exe_time = 0
        #Count the number of staying in the same state
        stay_num = 0
        e = 1.0 / 10000
        while Temp >= 1 and self.max_time - exe_time > e:
            next_solution = utils.random_switch(curr_solution)
            next_dist = utils.cal_total_dist(next_solution, self.dist_mat)
            if next_dist < curr_dist :
                curr_solution, curr_dist = next_solution, next_dist
            else:
                if np.random.uniform(0, 1) <= math.exp((self.best_dist - curr_dist)/Temp) :
                    curr_solution, curr_dist = next_solution, next_dist
                stay_num += 1
            if curr_dist < self.best_dist:
                self.best_solution, self.best_dist = curr_solution, curr_dist
                trace_time = time.time() - start_time
                self.trace.append([trace_time, int(self.best_dist)])
            if stay_num >= 100:
                stay_num, curr_solution, curr_dist = self.RestartToBest()
            Temp *= (1 - self.cooling_rate)
            exe_time = time.time() - start_time
    def Simulated_Annealing(self):
        #Set seed for the randomness
        np.random.seed(self.seed)
        #Generate random solution for the problem
        self.SA_process(self.Temp)
        return self.best_dist, self.best_solution, self.trace


if __name__ == '__main__':
    # Temporarily use Atlanta as the example

    # TSP_Data = parse("DATA/Atlanta.tsp")
    # test = LS1_SA(TSP_Data.to_adjacency_mat(), 0.001, np.random.randint(1, 10000))
    # solution = test.Simulated_Annealing()
    print("Using Simulated Annealing to solve TSP Probelm\n")
    

