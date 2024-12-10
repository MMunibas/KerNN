import torch
import torch.nn as nn
import numpy as np
import ase
from ase.io import read

# utils imports
from utils.plot_corr import plot_corr
from utils.neuralnets.FFNet import FFNet

'''
Calculator for the atomic simulation environment (ASE)
that evaluates energies and forces using a neural network
'''
class KerNNCalculator:

    #most parameters are just passed to the neural network
    def __init__(self,
                 model_path,                     #ckpt file from which to restore the model (can also be a list for ensembles)
                 atoms,                          #ASE atoms object
                 n_input,                        #len of descriptor
                 n_hidden,                       #number of hidden nodes
                 n_out):                      #number of outputs 

        #save checkpoint
        self.model_path = model_path
        

        #create neural network
        self.n_atoms = len(atoms)
        # build/define neural network architecture
        # build/define neural network architecture
        self.model = FFNet(n_input,n_hidden,n_out)
                              
        #load parameters
        self.model.load_state_dict(torch.load(model_path))
        self.model = self.model.double()

        #calculate properties once to initialize everything
        self._calculate_all_properties(atoms)
        
    def _calculate_all_properties(self, atoms):

        
        #standardize input features
        meanE = -0.42695848
        stdE  =  0.86893309
        
        minr = torch.from_numpy(np.array([2.12126003, 1.02126003, 1.10000000]))
        meanK = torch.from_numpy(np.array([0.00041426, 0.00099777, 0.00378366]))
        stdK  = torch.from_numpy(np.array([0.00098933, 0.00496903, 0.01045940]))


        pos = torch.tensor(atoms.get_positions(), requires_grad=True)#.double()
        self.nintdist = int(self.n_atoms *(self.n_atoms -1) /2)
        R = self._get_bond_length(pos)

        
        k = (self._get_1D_kernels_k33(R, minr) - meanK) / stdK


        #calculate energy and forces
        self._last_energy = self.model(k) * stdE + meanE
        
        self._last_forces = -torch.autograd.grad(torch.sum(self._last_energy), pos, create_graph=True)[0]


        #store copy of atoms
        self._last_atoms = atoms.copy()
        
        
        
        
    def _get_bond_length(self, pos):
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
            dist = torch.zeros(self.nintdist)
            dist[0] = torch.linalg.norm(pos[0, :] - pos[1, :])
            dist[1] = torch.linalg.norm(pos[1, :] - pos[2, :])
            dist[2] = torch.linalg.norm(pos[0, :] - pos[2, :])
        elif len(pos.shape) == 3:
            dist = torch.zeros(pos.shape[0], self.nintdist)
            dist[:, 0] = torch.linalg.norm(pos[:, 0, :] - pos[:, 1, :], axis=1)
            dist[:, 1] = torch.linalg.norm(pos[:, 1, :] - pos[:, 2, :], axis=1)
            dist[:, 2] = torch.linalg.norm(pos[:, 0, :] - pos[:, 2, :], axis=1)
        else:
            print("ERROR: Please check that the shape of the position array is correct")
        return dist

    def _get_1D_kernels_k33(self, x, xi, scale=1):
        """function calculating the 1D kernel functions given the bond length
        x corresponds to the input, xi is the reference structure)
        """
        xl = torch.maximum(x, xi)
        xs = torch.minimum(x, xi)

        drker33 = scale*(3.0 / (20.0 * xl ** 4) - 6.0 / 35.0 * xs / xl ** 5 + 3.0 / 56.0 * xs ** 2 / xl ** 6)
        return drker33
    
        

    def get_potential_energy(self, atoms, force_consistent=False):
        if self.calculation_required(atoms):
            self._calculate_all_properties(atoms)
        return self.last_energy.detach().numpy()

    def get_forces(self, atoms):
        if self.calculation_required(atoms):
            self._calculate_all_properties(atoms)
        return self.last_forces.detach().numpy() #the [0] is to prevent some "problems"
        
        
    def calculation_required(self, atoms, quantities=None):
        return atoms != self.last_atoms


        

    @property
    def sess(self):
        return self._sess

    @property
    def last_atoms(self):
        return self._last_atoms

    @property
    def last_energy(self):
        return self._last_energy[0]

    @property
    def last_forces(self):
        return self._last_forces
 
    @property
    def energy(self):
        return self._energy

    @property
    def forces(self):
        return self._forces


