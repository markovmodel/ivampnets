# iVAMPnets

Codebase for the iVAMPnets estimator and model which includes the classes for constructing the masks for toymodels and real protein applications.
The implemented methods allow to decompose a possible high dimensional system in its weakly coupled or independent subsystems. Thereby, the downstream estimation of the kinetic models is much more data efficient than estimating a global kinetic model which might not be feasible. The whole pipeline is an end-to-end deep learning framework which allows to define your own network architectures for the kinetics estimation of each subsystem. 
The data for the synaptotagmin C2A system is available upon request. The code is designed to reproduce the results of our paper "A deep learning framework for the decomposition of macromolecules into independent VAMPnets" (Link will be added) and is based on the deeptime package (see https://deeptime-ml.github.io/latest/index.html). 

The code includes:
1. (ivampnets.py) The definition of the ivampnets estimator class, which allows to fit a given model to simulation data. The definition of the ivampnets model class - the resulting model - which can then be used to estimate transition matrices, implied timescales, eigenfunctions, etc.
2. (masks.py) The definition of the mask modules, which can be used to give the modeler an intuition which part of the global system is assigned to which subsystem.
3. (examples.py) Helper functions to generate the data for the toy systems and plot some results.
4. (Toymodel_2Systems.ipynb) Notebook to reproduce the results for a simple truly independent 2D system. 
5. (10Cube.ipynb) Notebook to reproduce the results for the 10-Cube example.
6. (SynaptotagminC2A.ipynb) Notebook to reproduce the results for a protein example. The data of the synaptotagmin C2A domain is available upon request.

The code was executed using the following package versions:

python=3.6 or higher
jupyterlab=3.2.0 or jupyter=1.0.0

pytorch=1.8.0
deeptime=0.2.9
numpy=1.19.5
matplotlib=3.1.3

optional:
tensorboard=2.6.0
h5py=1.10.4

