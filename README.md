# Bayesian Controller Fusion

A hybrid control strategy for combining deep RL and classical robotic controllers. We provide two environments for both navigation and reaching tasks. For each task, we addtionally provide traditional handcrafted controllers that can solve part of the task, however are not the optimal solution.

### Installation

The only requirement is that you have [Conda installed](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) on your system, and [NVIDIA drivers installed](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&=Ubuntu&target_version=20.04&target_type=deb_network) if you want CUDA acceleration. To install all the required python dependencies run the following command within conda.

```
conda env create --file env_requirements.yml/
```

For the manipuliabity maximising reacher task, you will additionally require PyRep and CoppeliaSim. Head to the 
[PyRep github](https://github.com/stepjam/PyRep) page for installation instructions for this environment. 


### Usage

The complete training pipeline for BCF and the baselines compared in this work are provides in 'main.py'. The file allows for several input arguments to allow the user to specify the task, algorithm, prior controller, and the respective hyperparameters. 

```
git clone https://github.com/krishanrana/bcf.git
cd bcf
python3 main.py --task "navigation" --method "BCF" --prior_controller "APF" --sigma_prior 0.4 --num_agents 10
```

### Logging

All results are logged using [Weights and Biases](https://wandb.ai). An account and initial login is required to initialise logging as described on thier website.

### Citation

```
  @article{rana2021bayesian,
    title={Bayesian Controller Fusion: Leveraging Control Priors in Deep Reinforcement Learning for Robotics},
    author={Rana, Krishan and Dasagi, Vibhavari and Haviland, Jesse and Talbot, Ben and Milford, Michael and S{\"u}nderhauf, Niko},
    journal={arXiv preprint arXiv:2107.09822},
    year={2021}
  }
```


