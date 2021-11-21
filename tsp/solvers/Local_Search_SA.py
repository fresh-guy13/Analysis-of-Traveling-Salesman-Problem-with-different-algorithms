import sys
#If u wanted to run this file, run it under The Project Directory
sys.path.append("./tsp/")
from parse import parse
import utils
#from Project.tsp.graph import TspData
import numpy as np
#from tsp.parse import parse
import math
class Simulated_Annealing(object):
    def __init__(self, dist_mat, cooling_rate, seed):
        self.dist_mat = dist_mat
        self.candidate_array = np.asarray([i for i in range(1,len(self.dist_mat) + 1)])
        self.cooling_rate = cooling_rate
        self.best_solution = np.copy(self.candidate_array)
        self.best_dist = utils.cal_total_dist(self.best_solution, self.dist_mat)
        self.Temp = 1000
        self.seed = seed
        
    def SA_process(self, Temp):
        curr_solution = utils.gen_random_ans(self.candidate_array)
        curr_dist = utils.cal_total_dist(curr_solution, self.dist_mat)
        while Temp >= 1:
            next_solution = utils.random_switch(curr_solution)
            next_dist = utils.cal_total_dist(curr_solution, self.dist_mat)
            if next_dist < curr_dist :
                curr_solution = next_solution
                curr_dist = next_dist
            else:
                if np.random.uniform(0, 1) <= math.exp((self.best_dist - curr_dist)/Temp) :
                    curr_solution = next_solution
                    curr_dist = next_dist
            if curr_dist < self.best_dist:
                self.best_solution = curr_solution
                self.best_dist = curr_dist
            Temp *= (1 - self.cooling_rate)
    def Simulated_Annealing(self):
        #Set seed for the randomness
        np.random.seed(self.seed)
        #Generate random solution for the problem
        self.SA_process(self.Temp)
        return self.best_dist


        



if __name__ == '__main__':
    # Temporarily use Atlanta as the example
    #TODO:: Use tsp_main to parse the data
    TSP_Data = parse("DATA/Atlanta.tsp")
    test = Simulated_Annealing(TSP_Data.to_adjacency_mat(), 0.001, np.random.randint(1, 10000))
    solution = test.Simulated_Annealing()
    print(solution)
    

