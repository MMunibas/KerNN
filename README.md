<p align="center">
Pytorch implementation of KerNN<br>
Meuwly Group, University of Basel
</p>

# General
The present repository provides access to a Pytroch implementation of the KerNN codes[1], which can be used to learn the potential energy surface (PES) of molecular systems. Such PESs can be used to drive molecular dynamics (MD) simulations, which have long been a cornerstone of research in fields ranging from chemistry and biology to materials science and drug discovery. Central to the success of MD simulations is the overall quality of the PES, and two approaches that have been used very successfully in the past are traditional force fields (FFs) and machine learning (ML) potentials - each of which coming with very particular advantages and disadvantages. FFs are very efficient and allow to elucidate the dynamics of large systems, while ML potentials very accurately interpolate the data their trained on. In terms of speed, ML potentials are located between explicit *ab initio* calculations and FFs. The goal of KerNN is to reduce the computational cost of ML potentials, while retaining their accuracy.


This repository contains instructions for the installation and dependencies, which is followed by examples on training and using the neural network-based PESs. The following tutorial is based on H<sub>2</sub>CO; however, the same steps (data sets, training scripts, trained models, etc. are provided as well) can be followed for the other two systems that are described in detail in Reference [1].


# Installations & Dependencies

The following installation steps were tested on a Ubuntu 20.04 workstation and using
Conda 23.11.0 (see https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html). The installation of the dependencies typically takes less than 5 min. Occasional problems were encountered with the conda/miniconda installation, which required adding additional information to the .bashrc file (typically these lines are added automatically when installing conda) - see anac_block.txt for exemplary lines to add to bashrc.

a) If not installed already, install Miniconda on your machine (see https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

b) Create an environment named (e.g.) kernn_env, install Python 3.8:

    conda create --name kernn_env python=3.8

   Activate it:

    conda activate kernn_env

    (deactivating it by typing: conda deactivate)


c) With activated environment, all dependencies can be installed.

    pip install -r requirements.txt
    
d) Add the *utils* folder to your python path (to make it "visible" from everywhere). This requires you to add the following line to your .bashrc file. (adapt the path to your location).

    export PYTHONPATH=$HOME/phd_projects/KerNN/PYTORCH/github/KerNN/


e) If required, install Grimme's xtb-python software (e.g. to perfrom an initial sampling)

    conda config --add channels conda-forge
    conda install xtb-python




# Examples
### Training KerNN for H<sub>2</sub>CO.

The *ab initio* reference data for H<sub>2</sub>CO was available from previous work [2]. The data set contains a total of 4001 configurations generated using normal mode sampling [3] including the optimized H<sub>2</sub>CO structure. *Ab initio* energies, forces and dipole moments were obtained for all structures at the CCSD(T)-F12B/aug-cc-pVTZ-F12 level of theory using MOLPRO. To capture the equilibrium, room temperature, and higher energy regions of the PES, the normal mode sampling was carried out at eight different temperatures (10, 50, 100, 300, 500, 1000, 1500, and 2000 K). For each temperature, 500 structures were generated. The data set is given in *h2co/training/datasets* or is available from Zenodo (https://zenodo.org/records/3923823). The following steps can be done for any of the H<sub>2</sub>CO models described in Reference [1].

Most of the hyperparameters and settings for training are given in the *train_kernn_gpu.py* script. These include NN architecture (number of inputs to the neural network, number of nodes in the hidden layers, number of outputs), training parameters (number of validation points, batch size, validation batch size, learning rate, weight factor for the force term in the loss function, ...).
The number of training points and a seed are given as command line arguments to the training script. For a given combination of training set size and seed (here 3200 and 42, respectively)
make sure you create a folder named *models_ntrain3200_seed42* beforehand. It will be used to save the KerNN models. To start training on the reference data, run

    ./train_kernn_gpu.py 3200 42
    
The expected runtime for 100 epochs using a training set of 3200 is less than 30 seconds (running on CPU of a Intel(R) Core(TM) i7-9700K CPU @ 3.60GHz workstation.) The 
progress of the training is printed to the console and can be visualized with tensorboard (*i.e. by typing tensorboard --logdir .*, if the training is run in the same directory).
Note that a new model is only saved if its validation loss is lower than any of the models before. Once the validation loss has not improved for a total of 2000 epochs, the training is terminated automatically. Once the training has terminated, the script will automatically plot the correlation of the reference and predicted energies (and save a .png to the models folder) and once the plot
is closed it will print the test set statistics (*e.g. MAE(E), RMSE(E), MAE(F), ...*), which will also be saved to a text file. Note that if no GPU is available, the script will will run on CPU (which for such small systems is not slower anyway). The *runs* folder that is created when you train a model is only used to visualize the progress of the training and can be deleted after completion.


If it is required to repeat the prediction of the test set, this can be done by running (the eval script in essence is a copy of the train script, but with a "train=False" statement and the path of the model to be evaluated specified).

    ./eval_kernn.py 3200 42
    
after you have (almost at the end of the file) specified the path and the name of the model you want to evaluate. Note that it is important that you train and evaluate the model with the same exact specifications of number of training points and seed (else you will not predict the TEST set).

Note on the different models one could train in *h2co/training/*

- *kernn_ns*: Model that is not permutationally invariant and predicts energies and forces.
- *kernn_ns_efmu*: Model that is not permutationally invariant and predicts energies, forces and dipole moments.
- *kernn_s*: Model that is permutationally invariant and predicts energies and forces.




### Evaluations in Python/ASE
Most Python scripts that are used to evaluate the KerNN PESs make use of the atomic simulation environment (ASE) [4] and are written in Python. It is important to get used to ASE, which has very good tutorials online (\url{https://wiki.fysik.dtu.dk/ase/tutorials/tutorials.html#ase}). Scripts on how to use the PESs that have been used throughout the evalulation of Reference [1] are given in the *evaluation* folder. These can for example be used to 

- predict the energy of a given structure in .xyz format (predict_mol.py)

    ./predict_mol.py -i h2co.xyz
    
- or to optimize a given structure in .xyz format (optimize.py)

    ./optimize.py -i h2co.xyz
    
- or to calculate the harmonic frequencies of an optimized molecule (ase_vibrations.py). The calculated frequencies can be compared to the ones published in Reference [1]

    ./ase_vibrations.py -i opt_h2co.xyz

- or to run a gas phase MD simulation (md_run.py) - you can visualize the trajectory with *ase gui* or by first converting the *.traj* file to *.xyz* format using *Traj2Xyz.py*

    ./md_run.py -i opt_h2co.xyz

If you want to use a PES that you trained yourself, this will require the following changes (it is important to note that at the moment this will be required for every new model you generated - although it could be automatized). 

1. Open the *KerNNCalculator/KerNNCalculator.py* script and look for lines 46 to 52, which require adaptation (since every data set has slightly different means, standard deviations, ...). It is easiest to get these for a particular n_train and seed from the *eval_kernn.py* script. Look for lines 113 and following and remove the comments to print the required quantities - then save them to the *KerNNCalculator.py* script.

2. Adapt the parameters of the ASE calculator in all python files (*predict_mol.py*, *optimize.py*, *ase_vibrations.py*, *md_run.py*). These include the path to the model you want to use, the number of input nodes, the number of nodes in the hidden layer and the number of output nodes.

To make sure you use the correct model, with the correct parameters you could print the positions and the energy for a particular test set molecule using the *eval_kernn.py* script. Using the positions you just printed, create a new *.xyz* file and predict the energy of that using the *predict_mol.py* script. Note that they are likely not to be exactly the same due to numerical errors.

The models and evaluation scripts are also available for hydrogen oxalate and HeH<sub>2</sub><sup>+</sup>.

### Extension to new systems
The following list outlines the steps that would need to be followed to apply KerNN to a new system. I usually recommend to start with a non-permutational invariant PES.

1. Decide on a system that you want to study.
2. Generate a sufficiently large reference data set using, *e.g.*, MD simulations, normal mode sampling, diffusion monte carlo, ...
3. Extend the *utils/distances.py* which is responsible to calculate the interatomic distances for your system, following the examples given for H<sub>2</sub>CO
4. In the *train_kernn_gpu.py* script import (line 21) and use the function you have just added to the *distances.py* script
5. Start the training as described above.
6. Start with a thorough evaluation



# References
[1] Silvan Käser, Debasish Koner, and Markus Meuwly "The Bigger the Better? Accurate Molecular Potential Energy Surfaces from Minimalist Neural Networks" https://arxiv.org/abs/2411.18121

[2] Silvan Käser, Debasish Koner, Anders S. Christensen, O. Anatole von Lilienfeld, and Markus Meuwly "Machine Learning Models of Vibrating H2CO: Comparing Reproducing Kernels, FCHL, and PhysNet"
J. Phys. Chem. A 2020, 124(42), 8853-8865, DOI: 10.1021/acs.jpca.0c05979

[3] Smith, J. S., Isayev, O., Roitberg, A. E. ANI-1: "An extensible neural network potential
with DFT accuracy at force field computational cost." Chem. Sci. 2017, 8, 3192–3203, https://doi.org/10.1039/C6SC05720A

[4] Ask Hjorth Larsen et al, "The atomic simulation environment—a Python library for working with atoms", 2017, J. Phys.: Condens. Matter, 29, 273002,  DOI 10.1088/1361-648X/aa680e

[5] Kejie Shao, Jun Chen, Zhiqiang Zhao, Dong H. Zhang "Communication: Fitting potential energy surfaces with fundamental invariant neural network" J. Chem. Phys. 145, 071101 (2016) https://doi.org/10.1063/1.4961454

# Contact

If you have any questions about the codes free to contact Silvan Kaeser (silvan.kaeser@unibas.ch) or Prof. Markus Meuwly (m.meuwly@unibas.ch)


