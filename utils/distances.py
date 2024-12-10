# torch imports
import torch
import torch.nn as nn



def get_bond_length_ABA(pos, nintdist):
    """
    function that calculates the interatomic distances
    of a molecule with ABA symmetry (such as HeH2+ or H2O)
    This function does not take permutational invariance
    into account.
    
    the numbering is (for the HeH2+ case)
    H:  0
    He: 1
    H:  2
    
    and bond distances
    0: H1-He
    1: H2-He
    2: H1-H2
    
    """
    if len(pos.shape) == 2:
        dist = torch.zeros(nintdist)
        dist[0] = torch.linalg.norm(pos[0, :] - pos[1, :])
        dist[1] = torch.linalg.norm(pos[1, :] - pos[2, :])
        dist[2] = torch.linalg.norm(pos[0, :] - pos[2, :])
    elif len(pos.shape) == 3:
        dist = torch.zeros(pos.shape[0], nintdist)
        dist[:, 0] = torch.linalg.norm(pos[:, 0, :] - pos[:, 1, :], axis=1)
        dist[:, 1] = torch.linalg.norm(pos[:, 1, :] - pos[:, 2, :], axis=1)
        dist[:, 2] = torch.linalg.norm(pos[:, 0, :] - pos[:, 2, :], axis=1)
    else:
        print("ERROR: Please check that the shape of the position array is correct")
    return dist




def get_bond_length_ABCC(pos, nintdist):
    """
    function that calculates the interatomic distances
    of the h2co molecule given the cartesian coord.
    This function does not take permutational invariance
    into account.
    
    numbering is 
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
    if len(pos.shape) == 2:

        dist = torch.zeros(nintdist)
        dist[0] = torch.linalg.norm(pos[0, :] - pos[1, :])
        dist[1] = torch.linalg.norm(pos[0, :] - pos[2, :])
        dist[2] = torch.linalg.norm(pos[0, :] - pos[3, :])
        dist[3] = torch.linalg.norm(pos[1, :] - pos[2, :])
        dist[4] = torch.linalg.norm(pos[1, :] - pos[3, :])
        dist[5] = torch.linalg.norm(pos[2, :] - pos[3, :])
    elif len(pos.shape) == 3:
        dist = torch.zeros(pos.shape[0], nintdist)
        dist[:, 0] = torch.linalg.norm(pos[:, 0, :] - pos[:, 1, :], axis=1)
        dist[:, 1] = torch.linalg.norm(pos[:, 0, :] - pos[:, 2, :], axis=1)
        dist[:, 2] = torch.linalg.norm(pos[:, 0, :] - pos[:, 3, :], axis=1)
        dist[:, 3] = torch.linalg.norm(pos[:, 1, :] - pos[:, 2, :], axis=1)
        dist[:, 4] = torch.linalg.norm(pos[:, 1, :] - pos[:, 3, :], axis=1)
        dist[:, 5] = torch.linalg.norm(pos[:, 2, :] - pos[:, 3, :], axis=1)
    else:
        print("ERROR: Please check that the shape of the position array is correct")
    return dist

    
    

def get_bond_length_hoxa(pos, nintdist):
    """
    function that calculates the interatomic distances
    of the hydrogen oxalate molecule given the cartesian coord.
    numbering is 
    C:0
    C:1
    O:2
    O:3
    O:4
    O:5
    H:6
    and bond distances
    0: C0-C1
    1: C0-O2
    2: C0-O3
    3: C0-O4
    4: C0-O5
    5: C0-H
    
    6: C1-O2
    7: C1-O3    
    8: C1-O4    
    9: C1-O5   
   10: C1-H
   
   11: O2-O3
   12: O2-O4
   13: O2-O5
   14: O2-H
   
   15: O3-O4
   16: O3-O5
   17: O3-H
   
   18: O4-O5
   19: O4-H
    
   20: O5-H
    
    """
    if len(pos.shape) == 2:

        dist = torch.zeros(nintdist)
        dist[0] = torch.linalg.norm(pos[0, :] - pos[1, :])
        dist[1] = torch.linalg.norm(pos[0, :] - pos[2, :])
        dist[2] = torch.linalg.norm(pos[0, :] - pos[3, :])
        dist[3] = torch.linalg.norm(pos[0, :] - pos[4, :])
        dist[4] = torch.linalg.norm(pos[0, :] - pos[5, :])
        dist[5] = torch.linalg.norm(pos[0, :] - pos[6, :])
        
        dist[6] = torch.linalg.norm(pos[1, :] - pos[2, :])
        dist[7] = torch.linalg.norm(pos[1, :] - pos[3, :])
        dist[8] = torch.linalg.norm(pos[1, :] - pos[4, :])
        dist[9] = torch.linalg.norm(pos[1, :] - pos[5, :])
        dist[10] = torch.linalg.norm(pos[1, :] - pos[6, :])
        
        dist[11] = torch.linalg.norm(pos[2, :] - pos[3, :])
        dist[12] = torch.linalg.norm(pos[2, :] - pos[4, :])
        dist[13] = torch.linalg.norm(pos[2, :] - pos[5, :])
        dist[14] = torch.linalg.norm(pos[2, :] - pos[6, :])
        
        dist[15] = torch.linalg.norm(pos[3, :] - pos[4, :])
        dist[16] = torch.linalg.norm(pos[3, :] - pos[5, :])
        dist[17] = torch.linalg.norm(pos[3, :] - pos[6, :])
        
        dist[18] = torch.linalg.norm(pos[4, :] - pos[5, :])
        dist[19] = torch.linalg.norm(pos[4, :] - pos[6, :])

        dist[20] = torch.linalg.norm(pos[5, :] - pos[6, :])
        
        
    elif len(pos.shape) == 3:
        dist = torch.zeros(pos.shape[0], nintdist)
        dist[:, 0] = torch.linalg.norm(pos[:, 0, :] - pos[:, 1, :], axis=1)
        dist[:, 1] = torch.linalg.norm(pos[:, 0, :] - pos[:, 2, :], axis=1)
        dist[:, 2] = torch.linalg.norm(pos[:, 0, :] - pos[:, 3, :], axis=1)
        dist[:, 3] = torch.linalg.norm(pos[:, 0, :] - pos[:, 4, :], axis=1)
        dist[:, 4] = torch.linalg.norm(pos[:, 0, :] - pos[:, 5, :], axis=1)
        dist[:, 5] = torch.linalg.norm(pos[:, 0, :] - pos[:, 6, :], axis=1)

        dist[:, 6] = torch.linalg.norm(pos[:, 1, :] - pos[:, 2, :], axis=1)
        dist[:, 7] = torch.linalg.norm(pos[:, 1, :] - pos[:, 3, :], axis=1)
        dist[:, 8] = torch.linalg.norm(pos[:, 1, :] - pos[:, 4, :], axis=1)
        dist[:, 9] = torch.linalg.norm(pos[:, 1, :] - pos[:, 5, :], axis=1)
        dist[:, 10] = torch.linalg.norm(pos[:, 1, :] - pos[:, 6, :], axis=1)
        
        dist[:, 11] = torch.linalg.norm(pos[:, 2, :] - pos[:, 3, :], axis=1)
        dist[:, 12] = torch.linalg.norm(pos[:, 2, :] - pos[:, 4, :], axis=1)
        dist[:, 13] = torch.linalg.norm(pos[:, 2, :] - pos[:, 5, :], axis=1)
        dist[:, 14] = torch.linalg.norm(pos[:, 2, :] - pos[:, 6, :], axis=1)
        
        dist[:, 15] = torch.linalg.norm(pos[:, 3, :] - pos[:, 4, :], axis=1)
        dist[:, 16] = torch.linalg.norm(pos[:, 3, :] - pos[:, 5, :], axis=1)
        dist[:, 17] = torch.linalg.norm(pos[:, 3, :] - pos[:, 6, :], axis=1)
        
        dist[:, 18] = torch.linalg.norm(pos[:, 4, :] - pos[:, 5, :], axis=1)
        dist[:, 19] = torch.linalg.norm(pos[:, 4, :] - pos[:, 6, :], axis=1)
        
        dist[:, 20] = torch.linalg.norm(pos[:, 5, :] - pos[:, 6, :], axis=1)
    else:
        print("ERROR: Please check that the shape of the position array is correct")
    return dist




