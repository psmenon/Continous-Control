# Continous-Control RL

## Introduction

In this project I trained a double-jointed arm agent to follow a target location.

![alt text](https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif)

A reward of +0.1 is provided for each step that the agent's hands is in the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The task is episodic. In order to solve the environment, the agent must get an average score of +30 over 100 consecutive episodes.

## Training in Linux

1. Download the environment:  

    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
    
To train the agent on Amazon Web Services (AWS), and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) to obtain the environment.

(1) Create and activate a Python 3.6 environment using Anaconda:
   
   	conda create --name name_of_environment python=3.6
	source activate name_of_environment

(2) Clone repository and install dependencies

```bash
git clone https://github.com/psmenon/Continous-Control.git
cd python
pip install .
```

(3) Place the environment file Reacher_Linux.zip in the p2_continous-control/ folder and unzip the file:

```bash
$ unzip Reacher_Linux.zip
```

(4)  Launch Jupyter notebook

```bash
$ jupyter notebook
```
## Files

```bash
TD3_agent.py - contains the Agent .

model.py -  contains the Pytorch neural network (actor and critic).

Continuous_Control.ipynb -  contains the code.

checkpoint_Actor.pth -  contains the weights of the solved actor network

checkpoint_Critic1.pth - contains the weights of the solved critic1 network

checkpoint_Critic2.pth - contains the weights of the solved critic2 network

Report.pdf - the project report
```
