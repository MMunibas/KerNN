#!/usr/bin/env python3

# imports
import argparse
from ase import Atoms
from ase.io import read, write
from ase.optimize import *
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, ZeroRotation, Stationary
from ase.md.langevin import Langevin
from ase.io.trajectory import Trajectory
from os.path import splitext
from ase.md.verlet import VelocityVerlet
from ase.visualize import view
import numpy as np

from xtb.ase.calculator import XTB
#parse command line arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser._action_groups.pop()
required = parser.add_argument_group("required arguments")
required.add_argument("-i", "--input",   type=str,   help="input xyz",  required=True)
optional = parser.add_argument_group("optional arguments")
optional.add_argument("--label",   type=str,   help="prefix of calculator files",  default="calc_xtb1/md")
optional.add_argument("--charge",  type=float, help="total charge", default=1.0)
optional.add_argument("--magmom",  type=int, help="magnetic moment (number of unpaired electrons)", default=1)

args = parser.parse_args()
filename, extension = splitext(args.input)


#read input file 
atoms = read(args.input)


#setup calculator
calc = XTB(
    label=args.label,
    charge=args.charge,
    magmom=args.magmom)



#setup calculator (which will be used to describe the atomic interactions)
atoms.set_calculator(calc)                   

if args.charge==1:
    atoms.set_initial_charges([0, 1, 0, 0])    
    
#run an optimization
BFGS(atoms).run(fmax=0.0001)


# save final structure in xyz format
mol = atoms.copy() #make a copy, because else it saves forces as well - probably there is a more elegant way to circumvent that
write("opt_" + args.input, mol)




