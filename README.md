# iVAMPnets

Codebase for the iVAMPnets estimator and model which includes the classes for constructing the masks for toymodels and real protein applications.
The implemented methods allow to decompose a possible high dimensional system in its weakly coupled or independent subsystems. Thereby, the downstream estimation of the kinetic models is much more data efficient than estimating a global kinetic model which might not be feasible. The whole pipeline is an end-to-end deep learning framework which allows to define your own network architectures for the kinetics estimation of each subsystem. 
The data for the synaptotagmin C2A system is available upon request. The code is designed to reproduce the results of our paper "Deep learning to decompose macromolecules into independent Markovian domains" (https://www.biorxiv.org/content/10.1101/2022.03.30.486366v1) and is based on the deeptime package (see https://deeptime-ml.github.io/latest/index.html). 

The code includes:
1. (ivampnets.py) The definition of the ivampnets estimator class, which allows to fit a given model to simulation data. The definition of the ivampnets model class - the resulting model - which can then be used to estimate transition matrices, implied timescales, eigenfunctions, etc.
2. (masks.py) The definition of the mask modules, which can be used to give the modeler an intuition which part of the global system is assigned to which subsystem.
3. (examples.py) Helper functions to generate the data for the toy systems and plot some results.
4. (Toymodel_2Systems.ipynb) Notebook to reproduce the results for a simple truly independent 2D system. Typical runtime (cpu): 2 min
5. (10Cube.ipynb) Notebook to reproduce the results for the 10-Cube example. Typical runtime (cpu): 5 min
6. (SynaptotagminC2A.ipynb) Notebook to reproduce the results for a protein example. The data of the synaptotagmin C2A domain is available upon request. Typical runtime (cuda): 1.5 hours

The code was executed using the following package versions on a linux computer (debian bullseye):

```
python=3.6 or higher
jupyterlab=3.2.0 or jupyter=1.0.0

pytorch=1.8.0
deeptime=0.2.9
numpy=1.19.5
matplotlib=3.1.3
```
optional:
```
tensorboard=2.6.0
h5py=1.10.4
```

## Installation instructions

The software dependencies can be installed with anaconda / miniconda. If you do not have miniconda or anaconda, please follow the instructions here: https://conda.io/miniconda.html

The following command can be used to create a new conda environment and install all dependencies for the ivampnets scripts. 
```bash
conda create -n ivampnets pytorch=1.8.0 deeptime=0.2.9 numpy=1.19.5 matplotlib=3.1.3 jupyter h5py -c conda-forge
```
The new conda environment can be activated with
```bash
conda activate ivampnets
```


In case you are already a conda and jupyter notebook user with various environments, you can install your environment Python kernel via
```bash
python -m ipykernel install --user --name ivampnets
```
This repository including the python scripts and jupyter notebooks can be downloaded with 
```bash
git clone git@github.com:markovmodel/ivampnets.git
```

The following command will start the jupyter notebook server:
```bash
jupyter notebook
```

Your browser should pop up pointing to a list of notebooks once you navigate into the repository directory. If it's the wrong browser, add for example `--browser=firefox` or copy and paste the URL into the browser of your choice.

The typical install time ranges from 5 minutes for conda-users to 20 minutes if conda has to be set up from scratch.
