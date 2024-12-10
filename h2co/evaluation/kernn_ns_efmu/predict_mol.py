#!/usr/bin/env python3
import argparse
import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.optimize import *
from KerNNCalculator.KerNNCalculator import *

'''
Script to calculate the potential energy of a structure.
Usage: python predic_mol.py -i structure.xyz
'''

#parse command line arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser._action_groups.pop()
required = parser.add_argument_group("required arguments")
required.add_argument("-i", "--input",   type=str,   help="input xyz",  required=True)
optional = parser.add_argument_group("optional arguments")
optional.add_argument("--charge",  type=int, help="total charge", default=0)

args = parser.parse_args()
print("input ", args.input)

#read input file (molecular structure to predict) and create
#an atoms object
atoms = read(args.input)

#setup calculator object, which in this case is the NN calculator
#it is important that it is setup with the settings as used in the
#training procedure.
calc = KerNNCalculator(
    model_path="../final_models_h2co/nosym_3200_141_efmu/model_20240621_115220_233115_seed141_ema", #load the model you want to used
    atoms=atoms,
    n_input=6,
    n_hidden=20,
    n_out=5)

#attach the calculator object to the atoms object
atoms.set_calculator(calc)

#calculate properties of the molecule/geometry
#[for an extensive list see https://wiki.fysik.dtu.dk/ase/ase/atoms.html]
epot = atoms.get_potential_energy()
f = atoms.get_forces()
mu = np.dot(atoms.get_charges(), atoms.get_positions())#where atoms.get_charges() calculates
#the atomic partial charges (in elementary charges)

#print results
print("Epot = " + str(epot))#in eV
print("F = " + str(f)) #in eV/angstrom
print("charges = " + str(atoms.get_charges())) #in elementary charges
print("mu = " + str(mu)) #in elementary charges * ang



