# CSE 6140 Project

Final project for CSE 6140

Solvers for the Traveling Saleman Problem (TSP). Available solvers include:

1. Branch-and-bound
2. MST approximation
3. Local search using simulated annealing
4. Local search using 2-opt exchange neighborhood

# Installation

Requires Python 3.8 or higher.

Inside this repo you can run `pip3 install .`,  or `python3 -m pip install .` to install the code.
This will install the local python package `tsp`.

__Note__: This step is required since we're using Python bindings to C++ code.


# Directory Structure

All the code is located in the `tsp` directory. The overall structure is the following:

* `tsp/tsp_main.py` is the main script for running everything.
* `tsp/tsp_batch.py` is used for running multiple instances of an algorithm at various seeds.
* `tsp/solvers` contains each of the provided solvers.
* `tsp/solvers/src` provides a cpp implementation of branch-and-bound with python bindings.
* `tsp/scripts` provides scripts for plotting/evaluation of results.

# Running

The main script is `tsp/tsp_main.py`, and you can run it as so:

```
python3 tsp_main.py -inst <filename> -alg [BnB | Approx | LS1 | LS2] -time <cutoff_in_seconds> [-seed <random_seed>] [-odir <directory>]
```

Running this will result in two files being generated:

1. `<instance>_<method>_<cutoff>[_<random_seed>].sol`
   - Contains best distance and indices of best tour.
   - Example:
	 ```
      400
      [0, 1, 3, 2]
     ```
   
2. `<instance>_<method>_<cutoff>[_<random_seed>].trace`
   - Contains a timestamped log of the best tour distances as (distance, time) pairs.
   - Example:
     ```
     100, 0.1
     200, 0.2
     ```

