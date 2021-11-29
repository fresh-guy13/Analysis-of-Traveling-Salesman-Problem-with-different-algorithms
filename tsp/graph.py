"""
Graph related stuff
"""

from dataclasses import dataclass
import numpy as np
from typing import Callable, List, Tuple

Coord = Tuple[float, float]

@dataclass
class TspData:
    name: str
    comment: str
    dim: int
    edge_weight: Callable[[Coord], float]
    coords: np.ndarray

    def to_adjacency_mat(self):
        mat = np.zeros((self.dim, self.dim), dtype=np.uint32)
        for i in range(self.dim):
            for j in range(self.dim):
                mat[i,j] = self.edge_weight(self.coords[i], self.coords[j])

        return mat

