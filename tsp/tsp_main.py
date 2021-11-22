import argparse
from parse import parse
from solvers import Local_Search_SA
if __name__ == '__main__':

    # TODO: parse args and call solver
    parser = argparse.ArgumentParser()
    parser.add_argument('-city', type=str, dest='City', default='Atlanta', help='Takes the name of the city')
    parser.add_argument('-algo', type=str, dest='algorithm', default='BnB', help='The algorithm to solve the TSP problem')
    parser.add_argument('-seed', type=int, dest='seed', default=1, help='The number of the seed')
    parser.add_argument('-time', type=int, dest='maxtime', default=100, help='The cutoff time of the algorithm')
    args = parser.parse_args()
    #Get the data file
    data_path = '../DATA/{}.tsp'.format(args.City)
    #Get the predefined class of TSP data
    data = parse(data_path)
    dist_mateix = data.to_adjacency_mat()
    if args.algorithm == 'BnB':
        #TODO: Run Branch and Bound 
        print("Use the BnB algorithm")
    elif args.algorithm == 'Approx':
        #TODO: Run Approximate Solution
        print("Use the Approx algorithm")
    elif args.algorithm == 'LS1':
        #TODO: Run the LS1 (Simulated Annealing)Solution
        sol_instance = Local_Search_SA.LS1_SA(dist_mateix, 0.001, args.seed)
        solution = sol_instance.Simulated_Annealing()
        print("The min distance of {} is {}".format(args.algorithm, solution))
    elif args.algorithm == 'LS2':
        #TODO: Run the LS2 Solution
        print("Use the LS2 algorithm")
    else:
        print("The {} algorithm is not supported, please try another".format(args.algorithm))
