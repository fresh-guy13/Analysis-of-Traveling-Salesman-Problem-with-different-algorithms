#### Utils function for solvers###
import numpy as np

def gen_random_ans(arr):
    np.random.shuffle(arr)
    return arr

def cal_total_dist(arr, mat):
    total_dist = 0
    for i in range(mat.shape[0] - 1):
        total_dist += mat[arr[i] - 1][arr[i + 1] - 1]
    return total_dist

def random_switch(arr):
    idx = np.random.randint(0, len(arr))
    if idx == len(arr) - 1:
        arr[0], arr[idx] = arr[idx], arr[0]
    else:
        arr[idx], arr[idx + 1] = arr[idx + 1], arr[idx]
    return arr

def gen_solution_file(dist, arr, file_name):
    #arr is the 1D list of trace e.g [1,3,5,..]
    #dist is the solution found by the algorithm
    with open(file_name, "w") as file:
        file.writelines(str(dist)+'\n')
        file.writelines(str(arr[0] - 1))
        for i in arr[1:]:
            file.write(','+str(i-1))