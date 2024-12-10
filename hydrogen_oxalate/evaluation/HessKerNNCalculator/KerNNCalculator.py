import torch
import torch.nn as nn
import numpy as np
import ase
from ase.io import read

# utils imports
from utils.plot_corr import plot_corr
from utils.neuralnets.FFNet import FFNet
from utils.distances import get_bond_length_hoxa
from utils.kernels import get_1D_kernels_k33

# define options
torch.set_printoptions(precision=8)

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
        def forward_pass(pos):
        
            #standardize input features
            meanE = -37.92519379
            stdE  =   0.63330263

            minr = torch.from_numpy(np.array([1.58073997, 1.35041451, 1.21015096, 2.42493105, 2.35440564, 1.79517293,
                                                2.39583564, 2.51006699, 1.23457229, 1.27386451, 2.08413124, 2.25074625,
                                                3.52344871, 2.48022628, 1.00028789, 2.91033292, 3.50622654, 2.94421911,
                                                2.28182817, 3.31586099, 1.68671227]))

            meanK = torch.from_numpy(np.array([0.00502270, 0.00950028, 0.01489529, 0.00092570, 0.00103752, 0.00275138,
                                                0.00094902, 0.00079460, 0.01393065, 0.01231084, 0.00151021, 0.00125921,
                                                0.00022387, 0.00076347, 0.02860118, 0.00041700, 0.00022308, 0.00042611,
                                                0.00119358, 0.00027631, 0.00351638]))
                                                
                                                
            stdK  = torch.from_numpy(np.array([4.14728071e-04, 6.88844826e-04, 7.17454939e-04, 6.48989153e-05,
                                                6.59814468e-05, 3.70084221e-04, 8.13421721e-05, 5.20993126e-05,
                                                5.89049014e-04, 1.03325548e-03, 5.21751528e-04, 7.39908864e-05,
                                                3.77639371e-05, 1.79841998e-04, 7.17267394e-03, 5.02451148e-05,
                                                2.67221367e-05, 6.96967109e-05, 5.23924682e-05, 7.77031528e-05,
                                                2.23539094e-03]))


            self.nintdist = int(self.n_atoms *(self.n_atoms -1) /2)
            R = get_bond_length_hoxa(pos, self.nintdist)

            
            k = (get_1D_kernels_k33(R, minr) - meanK) / stdK


            #calculate energy and forces
            return self.model(k)[0] * stdE + meanE
            
        
        pos = torch.tensor(atoms.get_positions(), requires_grad=True)#.double()
        self._last_energy = forward_pass(pos)

 
        self._last_forces = -torch.autograd.grad(torch.sum(self._last_energy), pos, create_graph=True)[0]
        self._last_hessian = torch.autograd.functional.hessian(forward_pass, pos, create_graph=True)
        
        #store copy of atoms
        self._last_atoms = atoms.copy()
        
        
        
    
        

    def get_potential_energy(self, atoms, force_consistent=False):
        if self.calculation_required(atoms):
            self._calculate_all_properties(atoms)
        return self.last_energy.detach().numpy()

    def get_forces(self, atoms):
        if self.calculation_required(atoms):
            self._calculate_all_properties(atoms)
        return self.last_forces.detach().numpy() 

    def get_hessian(self, atoms):
        if self.calculation_required(atoms):
            self._calculate_all_properties(atoms)
        return self.last_hessian.detach().numpy() 

        
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
        return self._last_energy

    @property
    def last_forces(self):
        return self._last_forces
    
    @property
    def last_hessian(self):
        return self._last_hessian

    @property
    def energy(self):
        return self._energy

    @property
    def forces(self):
        return self._forces


    @property
    def hessian(self):
        return self._hessian

