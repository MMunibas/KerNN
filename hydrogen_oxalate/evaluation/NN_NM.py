#!/usr/bin/env python3


import argparse
import numpy as np
from ase import io
from ase.io.trajectory import Trajectory
from ase.io import read, write
from HessKerNNCalculator.KerNNCalculator import *
from os.path import splitext
import ase.units as units

'''
Usage: python3 NN_NM.py -i structure.xyz
'''

#parse command line arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser._action_groups.pop()
required = parser.add_argument_group("required arguments")
required.add_argument("-i", "--input",   type=str,   help="input traj",  required=True)
optional = parser.add_argument_group("optional arguments")
optional.add_argument("--charge",  type=int, help="total charge", default=-1)


args = parser.parse_args()
filename, extension = splitext(args.input)

atoms = read(args.input)
n_atom = len(atoms)

#setup calculator object, which in this case is the NN calculator
#it is important that it is setup with the settings as used in the
#training procedure.
calc = KerNNCalculator(
    model_path="models/model_20240712_152412_63811_seed42_ema", #load the model you want to used
    atoms=atoms,
    n_input=21,
    n_hidden=50,
    n_out=8)

#attach calc to atoms object
atoms.set_calculator(calc)

#get hessian
hessian = np.reshape(calc.get_hessian(atoms), (3*n_atom, 3*n_atom))

m = atoms.get_masses()
indices = range(len(atoms))
indices = np.asarray(indices)

#calculate normal mode freq.
im = np.repeat(m[indices]**-0.5, 3)
omega2, modes = np.linalg.eigh(im[:, None] * hessian * im)
modes = modes.T.copy()

# Conversion factor:
s = units._hbar * 1e10 / np.sqrt(units._e * units._amu)
hnu = s * omega2.astype(complex)**0.5



#print('  #    meV    cm^-1\n')
s = 0.01 * units._e / units._c / units._hplanck

for n, e in enumerate(hnu):
    if e.imag != 0:
        c = 'i'
        e = e.imag
    else:
        c = ' '
        e = e.real
    print('%3d   %7.2f%s' % (n, s * e, c))

pred_hf = []
for n, e in enumerate(hnu):
    if n > -2:
        pred_hf.append(s * e.real)
        
pred_hf = np.array(pred_hf)
print(pred_hf)
#exp = np.genfromtxt(args.exp)

np.savetxt("harm_"+filename+".dat", pred_hf)





