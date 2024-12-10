# torch imports
import torch
import torch.nn as nn


def ABCC_sym(k1d, nintdist):
    """
    function that includes permutational invariance for a
    molecular system with ABCC symmetrie (such as H2CO).
    
    Permutational invariance is included using fundamental
    invariants (https://doi.org/10.1063/1.4961454)
    
    The numbering is 
    C:0
    O:1
    H:2
    H:3
    and bond distances
    0: C-O
    1: C-H1
    2: C-H2
    3: O-H2
    4: O-H3
    5: H2-H3
    
    """

    if len(k1d.shape) == 1:
        
        sym = torch.zeros(nintdist+1)
        sym[0] = k1d[0]
        sym[1] = k1d[1] + k1d[2]
        sym[2] = k1d[3] + k1d[4]
        sym[3] = k1d[1]**2 + k1d[2]**2
        sym[4] = k1d[3]**2 + k1d[4]**2
        sym[5] = k1d[1]*k1d[3] + k1d[2]*k1d[4]
        sym[6] = k1d[5]
    elif len(k1d.shape) == 2:
        sym = torch.zeros(k1d.shape[0], nintdist+1)
        sym[:, 0] = k1d[:, 0]
        sym[:, 1] = k1d[:, 1] + k1d[:, 2]
        sym[:, 2] = k1d[:, 3] + k1d[:, 4]
        sym[:, 3] = k1d[:, 1]**2 + k1d[:, 2]**2
        sym[:, 4] = k1d[:, 3]**2 + k1d[:, 4]**2
        sym[:, 5] = k1d[:, 1]*k1d[:, 3] + k1d[:, 2]*k1d[:, 4]
        sym[:, 6] = k1d[:, 5]

    else:
        print("ERROR: Please check that the shape of the 1D Kernel array is correct")
    return sym
    
    

