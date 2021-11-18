"""
Provides functions for getting and registering edge weight functions

Note: This isn't really necessary since our project only uses EUC_2D
"""
import numpy as np
from typing import Callable


# Global dictionary for mapping between edge weight functions
_edge_weight_funcs = {}

def get_edge_weight_func(key: str):
    """
    Returns edge weight function associated with given key
    """
    if key not in _edge_weight_funcs:
        raise RuntimeError(f"{key} does not exist as an edge weight function")
    return _edge_weight_funcs[key]

def register(key: str, func: Callable):
    """
    Registers (key, func) as an available edge weight function
    """
    if key in _edge_weight_funcs:
        raise Warning(f"Overwriting {key} edge weight function")
    _edge_weight_funcs[key] = func


def euc_2d(a, b):
    """
    Calculates Euclidean 2D distance and rounds to closest int
    """
    assert(len(a) == len(b) == 2)
    return round(
        np.sqrt(
            (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
        )
    )

######################
# Register functions #
######################
register("EUC_2D", euc_2d)

    
