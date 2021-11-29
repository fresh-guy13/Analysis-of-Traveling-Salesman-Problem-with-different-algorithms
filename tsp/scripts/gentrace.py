"""
Generate dummy trace files
"""


import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description="Generate trace files to test plot script")
    parser.add_argument("opt", action="store", type=int, help="Optimum (min) solution of tsp instance")
    parser.add_argument('-t', action="store", type=int, dest="stoptime", help="Stopping time (default=100)", default=100)
    parser.add_argument('-k', type=int, default=1, action="store", dest="k", help="Number of traces to generate (default=1)")
    parser.add_argument("c", type=int, action="store", help="Maximum cost of a candidate solution")
    parser.add_argument('prefix', type=str, help="what to call the trace files. They will be saved as prefix0.trace, prefix1.trace ... etc.")

    args = parser.parse_args()
    

    opt = args.opt
    c = args.c
    k = args.k
    prefix = args.prefix
    stoptime = args.stoptime

    for j in range(k):
        gen_trace(opt, c, prefix + str(j) + ".trace", stoptime, j)


def gen_trace(opt, c, filename, stoptime, seed=None):
    rng = np.random.default_rng(seed)
    if not stoptime:
        stoptime = 100
    ctime = 0.0

    min_so_far = c + opt
    output = []
    with open(filename, 'w', newline='\n') as fout:
        while ctime < stoptime:
            ctime = ctime + rng.random()
            candidate = rng.integers(low=opt, high=c)
            min_so_far = min(min_so_far, candidate)
            output.append(", ".join([str(ctime), str(min_so_far)]) + "\n")
        fout.writelines(output)


if __name__ == "__main__":
    main()