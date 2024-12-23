#!/usr/bin/env python3
import argparse
import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.optimize import *
from HessKerNNCalculator.KerNNCalculator import *

'''
This optimization script needs the hessian and uses that information
to scale the gradients such that the optimization is always to the closest
critical point
Unfortunately, this is not very robust, so i recommend using it only when the structure
is already close to a minimum or transition state!

Usage: python3 -i structure.xyz -o opt_structure.xyz --fmax 0.00001
'''

#parse command line arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser._action_groups.pop()
required = parser.add_argument_group("required arguments")
required.add_argument("-i", "--input",   type=str,   help="input xyz",  required=True)
required.add_argument("-o", "--output",  type=str,   help="output xyz", required=True)
optional = parser.add_argument_group("optional arguments")
optional.add_argument("--label",  type=str,   help="prefix of calculator files",  default="calc_optts/opt")
optional.add_argument("--charge",  type=int, help="total charge", default=-1)
optional.add_argument("--magmom",  type=int, help="magnetic moment (number of unpaired electrons)", default=0)
optional.add_argument("--fmax",    type=float, help="optimizer tolerance",     default=0.00001)
optional.add_argument("--ds",    type=float, help="scaling for descent direction",     default=0.5)
args = parser.parse_args()
print("input ", args.input)
print("output", args.output)

#read input file
atoms = read(args.input)
atoms.rattle()

#setup calculator object, which in this case is the NN calculator
#it is important that it is setup with the settings as used in the
#training procedure.
calc = KerNNCalculator(
    model_path="models/model_20240712_152412_63811_seed42_ema", #load the model you want to used
    atoms=atoms,
    n_input=21,
    n_hidden=50,
    n_out=8)


atoms.set_calculator(calc)

#helper function for getting the hessian eigenvalues
def get_modes_and_omega():
    #get the hessian
    hessian = calc.get_hessian(atoms).reshape(3*len(atoms),3*len(atoms))

    #diagonalize  hessian
    omega2, modes = np.linalg.eigh(hessian)

    #sort such that the translations/rotations are the bottom six modes
    idx = np.argsort(np.abs(omega2))
    omega2 = omega2[idx]
    modes[:,:] = modes[:,idx]

    #transform normalmodes to Cartesian displacement vectors and normalize
    modes = modes.T 
    for i in range(3*len(atoms)): #normalize displacements
        modes[i,:] /= np.linalg.norm(modes[i,:])

    return omega2, modes

#perform optimization
step = 0
forces = atoms.get_forces().reshape(-1)
fmax = np.max(np.abs(forces))
omega2, modes = get_modes_and_omega()

while fmax > args.fmax:
    #project forces along eigenmodes and scale by curvature to normalize
    direction = np.zeros_like(forces)
    for i in range(6,3*len(atoms)): 
        direction += (modes[i,:]*np.dot(modes[i,:],forces))/omega2[i]
    #update positions and other quantities
    atoms.set_positions(atoms.get_positions() + args.ds*direction.reshape(-1,3))
    forces = atoms.get_forces().reshape(-1)
    fmax = np.max(np.abs(forces))
    omega2, modes = get_modes_and_omega()
    step += 1
    print(step, atoms.get_potential_energy(), fmax)   

#write output file
with open(args.output,"w") as f:
    f.write(str(len(atoms))+"\n")
    f.write("optimized structure\n")
    for xyz, symbol in zip(atoms.get_positions(), atoms.get_chemical_symbols()):
        f.write(' {0:2} {1} {2} {3}\n'.format(symbol, *xyz))
