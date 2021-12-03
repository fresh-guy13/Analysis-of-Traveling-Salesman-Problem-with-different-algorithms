"""
Make required plots from trace files

"""

import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib
import os
from cycler import cycler

def plot_rt_distribution(traces, opt, title="Runtime distribution", show=False, fname_prefix=None):
    # traces is a list of np.arrays, each representing a trace.
    # A trace is stored as a np.array with two columns.
    # The first column is the time in seconds,
    # and the second column is the best score seen at that time.

    # N is the total number of trials
    N = len(traces)
    lengths = [t.shape[0] for t in traces]
    success_indices = [np.searchsorted(-t[:,1], -opt) for t in traces]
    
    is_success = [success_index < length for length, success_index in zip(lengths, success_indices)]
    # is_success[j] is True iff trace j reached the optimum before terminating
    
    successful_runtimes = sorted([trace[idx,0] for idx, s, trace in zip(success_indices, is_success, traces) if s])
    # take only the successful trials and sort their runtimes
    
    M = len(successful_runtimes)
    print("{} runs successful out of {}".format(M, N))
    ygrid = [(j+1)/N for j in range(M)]
    
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

def trace_calculate_qualities(t, opt):
    t_q = np.copy(t)
    t_q[:,1] = (t_q[:,1] - opt) / opt
    return t_q

def plot_qrtd(traces, opt, qualities, title="Qualified RTD", show=False, fname_prefix=None):
    N = len(traces)
    traces_with_qualities = [trace_calculate_qualities(t, opt) for t in traces]
    lengths = [t.shape[0] for t in traces_with_qualities]
    qrtd_slices = []
    ygrids = []
    fig, ax = plt.subplots()

    plt.xlabel('CPU time (s)')
    plt.ylabel('P(solve)')
    plt.title(title)
    ax.set_prop_cycle(cycler('color', ['black', 'g', 'blue']) * cycler('linestyle', ['-', '--', ':', '-.']))
    max_runtime = 0.0
    min_runtime = 1000
    for q in qualities:
        success_indices = [np.searchsorted(-t[:,1], -q) for t in traces_with_qualities]
        is_success = [success_index < length for length, success_index in zip(lengths, success_indices)]
        # is_success[j] is True iff trace j reached the optimum before terminating
        successful_runtimes = sorted([trace[idx,0] for idx, s, trace in zip(success_indices, is_success, traces) if s])
        # take only the successful trials and sort their runtimes
    
        M = len(successful_runtimes)
        min_runtime = min(min_runtime, successful_runtimes[0])
        max_runtime = max(max_runtime, successful_runtimes[-1])
        #print("{} runs successful out of {}".format(M, N))
        ygrid = [(j+1)/N for j in range(M)]
        ygrids.append(ygrid)
        qrtd_slices.append(successful_runtimes)
    
    for successful_runtimes, ygrid, q in zip(qrtd_slices, ygrids, qualities):
        ax.plot(successful_runtimes + [max_runtime], ygrid + [ygrid[-1]], label="{:.1%}".format(q))
    ax.set_xscale('log')
    ax.set_xticks( np.geomspace(min_runtime, max_runtime, 15).round() )
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.legend()
    if show:
        # user asked for interactive plot
        plt.show()
        
    if fname_prefix:
        # save file as a PDF
        fig.savefig(fname_prefix + "_qrtdist.pdf")

def plot_sqd(traces, opt, times, title="Solution quality distribution", show=False, fname_prefix=None):
    N = len(traces)
    traces_with_qualities = [trace_calculate_qualities(t, opt) for t in traces]
    ygrid = [(j+1)/N for j in range(N)]
    sqd_slices = []
    fig, ax = plt.subplots()

    plt.xlabel('Solution quality (%)')
    plt.ylabel('P(solve)')
    plt.title(title)
    ax.set_prop_cycle(cycler('color', ['black', 'g', 'blue']) * cycler('linestyle', ['-', '--', ':', '-.']))

    for tm in times:
        time_indices = [np.searchsorted(t[:,0], tm) for t in traces_with_qualities]
        qualities = [100*t[min(tmidx, len(t)-1), 1] for tmidx, t in zip(time_indices, traces_with_qualities)]
        sqd_slices.append(sorted(qualities))
    
    for slice, tm in zip(sqd_slices, times):
        ax.plot(slice, ygrid, label="{} s".format(tm))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.legend()
    if show:
        # user asked for interactive plot
        plt.show()
        
    if fname_prefix:
        # save file as a PDF
        fig.savefig(fname_prefix + "_sqd.pdf")

def make_boxplot(data, title="Runtime box-plot", show=False, fname_prefix=None):
    N = len(data)

    fig, ax = plt.subplots()
    plt.ylabel('CPU time (s)')

    plt.title(title)
    runtimes = []
    labels = []
    for k, v in data.items():
        runtimes.append(v)
        labels.append(k[0] + ", " + k[1])

    plt.boxplot(runtimes, labels=labels)
    if show:
        # user asked for interactive plot
        plt.show()
        
    if fname_prefix:
        # save file as a PDF
        fig.savefig(fname_prefix + "_box.pdf")

def main():
    parser = argparse.ArgumentParser(description="Make plots from trace files")
    parser.add_argument('tracefiles', nargs='+', help='Trace files')
    parser.add_argument('-opt', action="store", type=int, dest="opt", help="Optimum value of problem")
    parser.add_argument('-interactive', action="store_true", dest="show", help="Display interactive plots")
    parser.add_argument('-prefix', action="store", dest="fname_prefix", help="Prefix of output files")
    parser.add_argument('-qrtd', action="store_true", dest="qrtd", help="Plot qualified RTD")
    parser.add_argument('-qualities', type=float, nargs="+", dest="qualities", help="List of solution qualities")
    parser.add_argument('-times', type=float, nargs="+", dest="times", help="List of cutoff times (for SQD plot)")
    parser.add_argument('-maxruntime', type=int, dest="maxruntime", help="Maximum runtime allowed for batch")
    parser.add_argument('-rtd', action="store_true", dest="rtd", help="Plot runtime distribution")
    parser.add_argument('-sqd', action="store_true", dest="sqd", help="Plot SQD")
    parser.add_argument('-boxplot', action="store_true", dest="boxplot", help="Make box plots of running times")
    parser.add_argument('-title', action="store", dest="title", help="Title of plot")
    args = parser.parse_args()

    traces = []
    cities = []
    algs = []
    boxplot_data_dict = {}
    for fname in args.tracefiles:
        trace = np.array(pd.read_csv(fname, names=['time', 'score']))
        if args.maxruntime:
            maxscore = np.max(trace[:,1])
            trace = np.append(trace, np.array([[args.maxruntime, maxscore]]), axis=0)
        traces.append(trace)
        basename = os.path.basename(fname)
        s = basename.split("_")
        cities.append(s[0])
        algs.append(s[1])
        if args.boxplot:
            rtime = np.max(trace[:,0])
            if (s[0], s[1]) not in boxplot_data_dict:
                boxplot_data_dict[(s[0], s[1])] = []
            boxplot_data_dict[(s[0], s[1])].append(rtime)
    # traces is a list of np.arrays with two columns each
    # Each row of each array is formatted as [time, score]
    
    opt = args.opt
    # print(options.show)
    if args.boxplot:
        make_boxplot(boxplot_data_dict, show=args.show, fname_prefix=args.fname_prefix)
    else:
        if not opt:
            print("Must provide optimum value")
            exit(0)
    if args.rtd:
        plot_rt_distribution(traces, opt, show=args.show, fname_prefix=args.fname_prefix, title=args.title)
    if args.qrtd and args.qualities:
        plot_qrtd(traces, opt, args.qualities, show=args.show, fname_prefix=args.fname_prefix, title=args.title)
    if args.sqd and args.times:
        plot_sqd(traces, opt, args.times, show=args.show, fname_prefix=args.fname_prefix, title=args.title)

if __name__ == '__main__':
    main()