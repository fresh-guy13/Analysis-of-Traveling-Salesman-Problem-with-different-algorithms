"""
Generate dummy trace files
"""


import numpy as np
from optparse import OptionParser

def main():
    usage = """usage: python3 tsp/scripts/gentrace.py OPT c k prefix
OPT: optimum value (minimum)
c: maximum cost of a candidate solution
k: number of traces to generate
prefix: what to call the trace files. They will be saved as
prefix0.trace, prefix1.trace ... etc."""
    parser = OptionParser(usage)
    parser.add_option('-t', "--stoptime", action="store", type="int", dest="stoptime")

    (options, args) = parser.parse_args()
    
    if len(args) != 4:
        parser.error("Wrong number of arguments")

    opt = int(args[0])
    c = int(args[1])
    k = int(args[2])
    prefix = args[3]
    

    for j in range(k):
        gen_trace(opt, c, prefix + str(j) + ".trace", options.stoptime, j)


def gen_trace(opt, c, filename, stoptime=100, seed=None):
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