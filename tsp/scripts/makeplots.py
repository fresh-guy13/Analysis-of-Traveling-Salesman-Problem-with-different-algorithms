"""
Make required plots from trace files

"""

import pandas as pd
import numpy as np
from optparse import OptionParser
import matplotlib.pyplot as plt

def plot_rt_distribution(traces, opt, title="Runtime distribution", show=False, fname_prefix=None):
    N = len(traces[0])
    print(N)
    lengths = [t.shape[0] for t in traces]
    success_indices = [np.searchsorted(-t[:,1], -opt) for t in traces]
    print(N, lengths, success_indices)
    is_success = [success_index < length for length, success_index in zip(lengths, success_indices)]
    # is_success[j] is True iff trace j reached the optimum before terminating
    
    successful_runtimes = sorted([trace[idx,0] for idx, s, trace in zip(success_indices, is_success, traces) if s])
    # take only the successful trials and sort their runtimes
    
    M = len(successful_runtimes)
    ygrid = [(j+1)/N for j in range(M)]
    print(successful_runtimes, ygrid)
    fig, ax = plt.subplots()
    ax.plot(successful_runtimes, ygrid, color='red')
    plt.xlabel('CPU time (s)')
    plt.ylabel('P(solve)')
    plt.title(title)
    
    if show:
        # user asked for interactive plot
        plt.show()
        
    if fname_prefix:
        # save file as a PDF
        fig.savefig(fname_prefix + "_rtdist.pdf")

def main():
    usage = "python3 tsp/scripts/makeplots.py [OPTIONS] file1.trace file2.trace ...\nExample: python3 plotting.py -s outfile -o 30 traces/*"
    parser = OptionParser(usage)
    parser.add_option('-o', action="store", type="int", dest="opt", help="Optimum value of problem")
    parser.add_option('-i', action="store_true", dest="show", help="Display interactive plot")
    parser.add_option('-s', action="store", dest="fname_prefix", help="Prefix of output file")
    
    (options, args) = parser.parse_args()
    traces = [np.array(pd.read_csv(fname, names=['time', 'score'])) for fname in args]
    # traces is a list of np.arrays with two columns each
    # Each row of each array is formatted as [time, score]
    
    if not options.opt:
        parser.error("I can't make a plot without the optimum value")
    opt = options.opt
    print(options.show)
    plot_rt_distribution(traces, opt, show=options.show, fname_prefix=options.fname_prefix)
    

if __name__ == '__main__':
    main()