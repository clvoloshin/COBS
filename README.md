# Caltech OPE Benchmarking Suite (COBS)

## Introduction

COBS is an Off-Policy Policy Evaluation (OPE) Benchmarking Suite. The goal is to provide fine experimental control to carefully tease out an OPE method's performance across many key conditions. 

We'd like to make this repo as useful as possible for the community. We commit to continual refactoring and code-review to make sure the COBS continues to serve its purpose. Help is always appreciated!

COBS is based on Empirical Study of Off Policy Policy Estimation paper (https://arxiv.org/abs/1911.06854). We have migrated from Tensorflow to Pytorch and made COBS generally more easy to use. For the original TF implementation and replication of the paper, please see the paper branch.

## Getting started

### Tutorial

To get started using the experimental tools see [Tutorial.ipynb](https://github.com/clvoloshin/COBS/blob/master/Tutorial.ipynb).

### Installation

Tested on python3.6+.
```
python3 -m venv cobs-env
source cobs-env/bin/activate
pip3 install -r requirements.txt
```

## Experiment Configuration

See an example experiment [configuration](https://github.com/clvoloshin/COBS/blob/master/cfgs/nn_example_cfg.json). The configuration contains two parts, the experiment section and the models section. The experiment section is used to instatiate the environment and general parameters. The models section is used to specify which methods to use and their specific parameters.
The experiment section looks like:
```
 "experiment": {
        "gamma": 0.98,                  # discount factor
        "horizon": 5,                   # horizon of the environment
        "base_policy": 0.8,             # Probability of deviation from greedy for base policy. 
                                        #   Note: This parameter means different things depending on the type of policy
        "eval_policy": 0.2,             # Probability of deviation from greedy for evaluation policy. 
                                        #   Note: This parameter means different things depending on the type of policy
        "stochastic_env": true,         # Make environment have stochastic transitions 
        "stochastic_rewards": false,    # Make environment have stochastic rewards
        "sparse_rewards": false,        # Make environment have sparse rewards
        "num_traj": 8,                  # Number of trajectories to collect from base_policy/behavior_policy (pi_b)
        "is_pomdp": false,              # Make the environment a POMDP
        "pomdp_horizon": 2,             # POMDP horizon, if POMDP is true
        "seed": 1000,                   # Seed
        "experiment_number": 0,         # Label for experiment. Used for distributed compute
        "access": 0,                    # Credentials for AWS. Used for distributed compute
        "secret": 0,                    # Credentials for AWS. Used for distributed compute
        "to_regress_pi_b": {
            "to_regress": false,        # Should we regress pi_b? Is it unknown?
            "model": "defaultCNN",      # What model to fit pi_b with
                                        #  Note: To add your own, see later in the README.md
            "max_epochs": 100,          # Max number of fitting iterations
            "batch_size": 32,           # Minibatch size
            "clipnorm": 1.0             # Gradient clip
        },
        "frameskip": 1,                 # (x_t, a, r, x_{t+frameskip}). Apply action "a" frameskip number of times
        "frameheight": 1                # (x_{t:t+frameheight}, a, r, x_{t+1:t+1+frameheight}). State is consider a concatenation of frameheight number of states
    },
```
and the models section (TODO: rename to methods section) looks like:
```
 "models": {
        "FQE": {
            "model": "defaultCNN",        # What model to fit FQE with
            "convergence_epsilon": 1e-4,  # When to stop iterations
            "max_epochs": 100,            # Max number of fitting iterations
            "batch_size": 32,             # Minibatch size
            "clipnorm": 1.0               # Gradient clip
        },
        "Retrace": {
            "model": "defaultCNN",
            "convergence_epsilon": 1e-4,
            "max_epochs": 3,
            "batch_size": 32,
            "clipnorm": 1.0,
            "lamb": 0.9                   # Lambda, parameter for this family of method
        },
        ...
```

## Environments

To add a new environment, implement an OpenAI gym-like environment and place the environment in [the envs directory](https://github.com/clvoloshin/COBS/tree/master/ope/envs). The environment should implement the reset, step, and (optionally) render functions. Each environment must also contain two variables
```
self.n_dim # The number of states (if discrete), otherwise set this to 1.
self.n_actions # The number of possible actions 
```

See [Tutorial.ipynb](https://github.com/clvoloshin/COBS/blob/master/Tutorial.ipynb) for how to instantiate the environment during an experiment.

## Baselines

### Direct Method

To add a new Direct Method, implement one of the [Direct Method classes](https://github.com/clvoloshin/COBS/blob/master/ope/algos/direct_method.py) and put the new method in the [algos directory](https://github.com/clvoloshin/COBS/blob/master/ope/algos).

#### Q Function Based
Suppose your new method is called NewMethod and it works by fitting a Q function.

Modify line 149 of [experiment.py](https://github.com/clvoloshin/COBS/blob/master/ope/experiment_tools/experiment.py) by adding:
```
...
elif 'NewMethod' == model:
    new_method = NewMethod() ## Instatiates the method
    new_method.fit(behavior_data, pi_e, cfg, cfg.models[model]['model']) ## Fits the method
    new_method_Qs = new_method.get_Qs_for_data(behavior_data, cfg) ## Gets Q(s, a) for each s in the data and a in the action space. 
    out = self.estimate(new_method_Qs, behavior_data, gamma, model, true) ## Get Direct and Hybrid estimates and error.
    dic.update(out)
...
```

#### Weight Function Based
Suppose your new method is called NewMethod and it works by fitting a weight function.

Modify line 149 of [experiment.py](https://github.com/clvoloshin/COBS/blob/master/ope/experiment_tools/experiment.py) by adding:
```
...
elif 'NewMethod' == model:
    new_method = NewMethod() ## Instatiates the method
    new_method.fit(behavior_data, pi_e, cfg, cfg.models[model]['model']) ## Fits the method
    new_method_output = new_method.evaluate(behavior_data, cfg) ## Evaluate the method
    dic.update({'IH': [new_method_output, (new_method_output - true )**2]}) ## Update results
...
```

### Hybrid Method

TODO

### IPS Method

TODO

## Adding a new model to the configuration

Add your own NN architechture to a new file in the [models directory](https://github.com/clvoloshin/COBS/tree/master/ope/models). Then modify the get_model_from_name function in [factory.py](https://github.com/clvoloshin/COBS/blob/master/ope/experiment_tools/factory.py)
```
from ope.models.YourNN import YourNN

def get_model_from_name(name):
    ...
    elif name == 'YourNN'
        return YourNN
    ...
```
You can now add your own NN as a method's model in the configuration:
```
"SomeMethod": {
            "model": "YourNN",        # YourNN model
            ...other params....
        },
``` 

## Policies

There are currently two available policy types.


## <a name="CitingCOBS"></a>Citing COBS

If you use COBS, please use the following BibTeX entry.

```
  @misc{voloshin2019empirical,
    title={Empirical Study of Off-Policy Policy Evaluation for Reinforcement Learning},
    author={Cameron Voloshin and Hoang M. Le and Nan Jiang and Yisong Yue},
    year={2019},
    eprint={1911.06854},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```


