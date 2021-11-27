import argparse
from pathlib import Path
from parse import parse
from solvers import Local_Search_SA, utils

def main():
    # TODO: parse args and call solver
    parser = argparse.ArgumentParser()
    parser.add_argument('-inst', type=str, dest='inst', default='../DATA/Atlanta.tsp', help='Takes the name of the city')
    parser.add_argument('-algo', type=str, dest='algorithm', default='BnB', help='The algorithm to solve the TSP problem')
    parser.add_argument('-seed', type=int, dest='seed', default=1, help='The number of the seed')
    parser.add_argument('-time', type=int, dest='maxtime', default=10, help='The cutoff time of the algorithm')
    parser.add_argument('-odir', type=str, dest='odir', default=".", help='Where to store output files')
    args = parser.parse_args()

    #Get the predefined class of TSP data using given data_path
    data = parse(args.inst)
    dist_matrix = data.to_adjacency_mat()
    if args.algorithm == "LS1" or args.algorithm == "LS2":
        sol_path = args.odir + "/{}_{}_{}_{}.sol".format(Path(args.inst).stem, args.algorithm, args.maxtime, args.seed)
        trace_path = args.odir + "/{}_{}_{}_{}.trace".format(Path(args.inst).stem, args.algorithm, args.maxtime, args.seed)
    else:
        sol_path = args.odir + "/{}_{}_{}.sol".format(Path(args.inst).stem, args.algorithm, args.maxtime)
        trace_path = args.odir + "/{}_{}_{}_{}.trace".format(Path(args.inst).stem, args.algorithm, args.maxtime, args.seed)

    if args.algorithm == 'BnB':
        #TODO: Run Branch and Bound 
        print("Use the BnB algorithm")
    elif args.algorithm == 'Approx':
        #TODO: Run Approximate Solution
        print("Use the Approx algorithm")
    elif args.algorithm == 'LS1':
        #TODO: Run the LS1 (Simulated Annealing)Solution
        sol_instance = Local_Search_SA.LS1_SA(dist_matrix, 0.00001, args.seed, args.maxtime)
        solution = sol_instance.Simulated_Annealing()
        print("The min distance of {} is {}".format(args.algorithm, solution[0]))
        utils.gen_solution_file(solution[0], solution[1], sol_path)
        utils.gen_trace_file(solution[2], trace_path)
    elif args.algorithm == 'LS2':
        #TODO: Run the LS2 Solution
        print("Use the LS2 algorithm")
    else:
        print("The {} algorithm is not supported, please try another".format(args.algorithm))
    
if __name__ == '__main__':
    main()