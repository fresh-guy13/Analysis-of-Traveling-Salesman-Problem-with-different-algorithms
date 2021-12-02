"""
Util functions for solvers
"""

import numpy as np
import random
from numba import jit, njit

def gen_random_ans(arr):
    np.random.shuffle(arr)
    return arr

@njit
def cal_total_dist(arr, mat):
    total_dist = 0
    for i in range(-1, mat.shape[0] - 1):
        total_dist += mat[arr[i] - 1][arr[i + 1] - 1]
    return total_dist

def random_switch(arr):
    arr_copy = np.copy(arr)
    idx = random.sample(range(0, len(arr)), 2)
    arr_copy[idx[0]], arr_copy[idx[1]] = arr_copy[idx[1]], arr_copy[idx[0]]
    # if idx == len(arr) - 1:
    #     arr_copy[0], arr_copy[idx] = arr[idx], arr[0]
    # else:
    #     arr_copy[idx], arr_copy[idx + 1] = arr[idx + 1], arr[idx]
    return arr_copy

def gen_solution_file(dist, arr, file_name):
    #arr is the 1D list of trace e.g [1,3,5,..]
    #dist is the solution found by the algorithm
    with open(file_name, "w") as file:
        file.writelines(str(dist)+'\n')
        file.writelines(str(arr[0]))
        for i in arr[1:]:
            file.write(','+str(i))

def gen_trace_file(time_dist, file_name):
    with open(file_name, "w") as file:
        for i in time_dist:
            file.write('{:.2f}, {}\n'.format(i[0], i[1]))
