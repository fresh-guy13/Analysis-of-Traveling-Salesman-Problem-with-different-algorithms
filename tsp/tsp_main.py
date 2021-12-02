import argparse
import logging
from pathlib import Path
from tsp.parse import parse
from tsp.solvers import BranchAndBound, LS1_SA, LS2_2opt, MSTApprox, utils

def main():
    """
    Parse args and run tsp_main
    """
    # TODO: parse args and call solver
    parser = argparse.ArgumentParser()
    parser.add_argument('-inst', type=str, dest='inst', default='../DATA/Atlanta.tsp', help='Path of input tsp file')
    parser.add_argument('-alg', type=str, dest='algorithm', default='BnB', help='The algorithm to solve the TSP problem')
    parser.add_argument('-seed', type=int, dest='seed', default=1, help='The number of the seed')
    parser.add_argument('-time', type=int, dest='maxtime', default=10, help='The cutoff time of the algorithm')
    parser.add_argument('-odir', type=str, dest='odir', default=".", help='Where to store output files')
    parser.add_argument('-debug', action='store_true', default=False, help="Whether to print debug statements")
    args = parser.parse_args()
    run_tsp_main(args.inst, args.algorithm, args.seed, args.maxtime, args.odir, args.debug)

    
def run_tsp_main(inst, algorithm, seed, maxtime, odir, debug=False):
    """
    Run selected solver with given parameters
    """
    # Get the predefined class of TSP data using given data_path
    data = parse(inst)
    dist_matrix = data.to_adjacency_mat()

    if algorithm != 'BnB':
        sol_path = f"{odir}/{Path(inst).stem}_{algorithm}_{maxtime}_{seed}.sol"
        trace_path = f"{odir}/{Path(inst).stem}_{algorithm}_{maxtime}_{seed}.trace"
    else:
        sol_path = f"{odir}/{Path(inst).stem}_{algorithm}_{maxtime}.sol"
        trace_path = f"{odir}/{Path(inst).stem}_{algorithm}_{maxtime}.trace"

    # Get solver
    if algorithm == 'BnB':
        print("Using branch and bound")
        solver = BranchAndBound(data, maxtime, debug=debug)
    elif algorithm == 'Approx':
        print("Using the Approx algorithm")
        solver = MSTApprox(dist_matrix, seed)
    elif algorithm == 'LS1':
        print("Using Local Search 1: Simulated Annealing")
        solver = LS1_SA(dist_matrix, 0.00001, seed, maxtime)
    elif algorithm == 'LS2':
        print("Using Local Search 2: Iterative 2-Opt exchange")
        niters = 10 # what's the best value here?
        solver = LS2_2opt(data, seed, maxtime, niters, debug=debug)
    else:
        print("The {} algorithm is not supported, please try another".format(algorithm))
        exit(0)

    # Solve using selected solver
    best_dist, best_tour, trace = solver.solve()
    print(f"The min distance of {algorithm} is {best_dist}")
    
    # Generate trace files
    utils.gen_solution_file(best_dist, best_tour, sol_path)
    utils.gen_trace_file(trace, trace_path)
    
    
if __name__ == '__main__':
    main()
