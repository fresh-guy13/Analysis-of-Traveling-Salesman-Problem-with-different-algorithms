import argparse
import sys
import random
from tsp_main import run_tsp_main

def main():
    # TODO: parse args and call solver
    parser = argparse.ArgumentParser()
    parser.add_argument('-inst', type=str, dest='inst', default=None, help='Path of input tsp file')
    parser.add_argument('-alg', type=str, dest='algorithm', default='BnB', help='The algorithm to solve the TSP problem')
    parser.add_argument('-batch-seed', type=int, dest='batch_seed', default=1, help='Random seed for batch run. Used to generate seeds for each individual run.')
    parser.add_argument('-batch-size', type=int, dest='batch_size', default=1, help='Number of runs in the batch')
    parser.add_argument('-time', type=int, dest='maxtime', default=10, help='The cutoff time of the algorithm')
    parser.add_argument('-odir', type=str, dest='odir', default=".", help='Where to store output files')
    args = parser.parse_args()

    if not args.inst:
        raise SystemExit("Input tsp file not supplied!")

    seedmax = 2**32 - 1
    random.seed(args.batch_seed)
    print("Running a(n) {} batch of size {}".format(args.algorithm, args.batch_size))
    for i in range(args.batch_size):
        iter_seed = random.randint(0, seedmax)
        print("Iteration {}, seed {}".format(i, iter_seed))
        run_tsp_main(args.inst, args.algorithm, iter_seed, args.maxtime, args.odir)


    
if __name__ == '__main__':
    main()