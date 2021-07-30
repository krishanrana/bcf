# Bayesian Controller Fusion

A hybrid control strategy for combining deep RL and classical robotic controllers

### Installation

The only requirement is that you have [Conda installed](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) on your system, and [NVIDIA drivers installed](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&=Ubuntu&target_version=20.04&target_type=deb_network) if you want CUDA acceleration. To install all the required python dependencies run the following command within conda.

```
conda env create --file env_requirements.yml/
```

For the manipuliabity maximising reacher tasks, you will require PyRep and CoppeliaSim. Head to the 
[PyRep github](https://github.com/stepjam/PyRep) page for installation instructions for this environment. 

