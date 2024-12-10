import torch
import torch.nn as nn
import numpy as np
import ase
from ase.io import read

# utils imports
from utils.plot_corr import plot_corr
from utils.neuralnets.FFNet import FFNet
from utils.distances import get_bond_length_ABCC
from utils.kernels import get_1D_kernels_k33
from utils.symmetrize import ABCC_sym
torch.set_default_tensor_type(torch.DoubleTensor)

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
        meanE = -15.92512104
        stdE  =  0.20039307
        
        minr = torch.from_numpy(np.array([1.20929149, 1.10203065, 1.10203065, 2.02403646, 2.02403646, 1.88028233]))
        meanK = torch.from_numpy(np.array([1.51319674e-02, 4.21633728e-02, 3.77951108e-03,
                            9.00232269e-04, 7.16039324e-06, 8.00725631e-05, 2.53537998e-03]))
        stdK  = torch.from_numpy(np.array([4.34972618e-04, 2.63528490e-03, 1.40970941e-04,
                            1.16574176e-04, 5.37697631e-07, 8.00376420e-06, 1.50988968e-04]))


        pos = torch.tensor(atoms.get_positions(), requires_grad=True)#.double()
        self.nintdist = int(self.n_atoms *(self.n_atoms -1) /2)
        R = get_bond_length_ABCC(pos, self.nintdist)
        
        ktmp = get_1D_kernels_k33(R, minr, 1)
        k = (ABCC_sym(ktmp, self.nintdist) - meanK) / stdK

        #calculate energy and forces
        self._last_energy = self.model(k) * stdE + meanE
        
        self._last_forces = -torch.autograd.grad(torch.sum(self._last_energy), pos, create_graph=True)[0]


        #store copy of atoms
        self._last_atoms = atoms.copy()
        
        
        

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


