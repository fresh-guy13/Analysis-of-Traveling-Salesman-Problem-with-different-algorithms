"""
Provides parser for TSP files
"""

import numpy as np
import re

from .edge_weight import get_edge_weight_func
from .graph import TspData

DIGIT = '-?\d+(\.\d+)?'

patterns = {
    'name': re.compile('NAME: (\w+)'),
    'comment': re.compile('COMMENT: ([\w ]+)'),
    'dim': re.compile('DIMENSION: (\d+)'),
    'edge_weight': re.compile('EDGE_WEIGHT_TYPE: (\w+)'),
    'coord': re.compile(f'[\d]+ ({DIGIT}) ({DIGIT})')
}
    

def parse(filename: str) -> TspData:
    """
    Parses TSP data from a file
    """
    
    with open(filename, 'r') as f:

        data = {}
        
        if m := patterns['name'].match(f.readline()):
            data['name'] = m.group(1)
        else:
            raise RuntimeError("File must begin with 'NAME: ...'")

        if m := patterns['comment'].match(f.readline()):
            data['comment'] = m.group(1)
        else:
            raise RuntimeError("Second line of file must be 'COMMENT: ...'")

        if m := patterns['dim'].match(f.readline()):
            data['dim'] = int(m.group(1))
        else:
            raise RuntimeError("Third line of file must be 'DIMENSION: ...'")

        if m := patterns['edge_weight'].match(f.readline()):
            data['edge_weight'] = get_edge_weight_func(m.group(1))
        else:
            raise RuntimeError("Fourth line of file must be 'EDGE_WEIGHT_TYPE: ...'")

        if m := re.match(r"NODE_COORD_SECTION", f.readline()):
            pass
        else:
            raise RuntimeError("Fifth line of file must be 'NODE_COORD_SECTION'")

        ndims = data['dim']
        coords = np.zeros((ndims, 2))
        for i in range(ndims):
            if m := patterns['coord'].match(f.readline()):
                coords[i] = [float(m.group(1)), float(m.group(3))]
            else:
                raise RuntimeError("Failed to parse coordinate number {i+1}")
        data['coords'] = np.array(coords)

    return TspData(**data)
            

if __name__ == '__main__':
    
    # Test to make sure all the files parse correctly
    import glob
    files = glob.glob("../DATA/*.tsp")
    
    for f in files:
        d = parse(f)
        assert(d.dim == len(d.coords))

    
