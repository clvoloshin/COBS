import numpy as np
import argparse
import json
from copy import deepcopy

from ope.envs.graph import Graph
from ope.policies.basics import BasicPolicy

from ope.experiment_tools.experiment import ExperimentRunner, analysis
from ope.experiment_tools.config import Config
from ope.experiment_tools.factory import get_model_from_name

def main(param):

    # replace string of model with model itself in the configuration.
    for method, parameters in param['models'].items():
        if parameters['model'] != 'tabular':
            param['models'][method]['model'] = get_model_from_name(parameters['model'])

    runner = ExperimentRunner()

    for N in range(5): 
        configuration = deepcopy(param['experiment']) # Make sure to deepcopy as to never change original
        configuration['num_traj'] = 8*2**N # Increase dataset size
        
        cfg = Config(configuration)

         # initialize environment with this configuration
        env = Graph(make_pomdp=cfg.is_pomdp,
                    number_of_pomdp_states=cfg.pomdp_horizon,
                    transitions_deterministic=not cfg.stochastic_env,
                    max_length=cfg.horizon,
                    sparse_rewards=cfg.sparse_rewards,
                    stochastic_rewards=cfg.stochastic_rewards)

        # set seed for the experiment
        np.random.seed(cfg.seed)

        # processor processes the state for storage
        processor = lambda x: x

        # absorbing state for padding if episode ends before horizon is reached
        absorbing_state = processor(np.array([env.n_dim - 1]))

        # Setup policies
        actions = [0, 1]
        pi_e = BasicPolicy(
            actions, [max(.001, cfg.eval_policy), 1 - max(.001, cfg.eval_policy)])
        pi_b = BasicPolicy(
            actions, [max(.001, cfg.base_policy), 1 - max(.001, cfg.base_policy)])

        # add env, policies, absorbing state and processor
        cfg.add({
            'env': env,
            'pi_e': pi_e,
            'pi_b': pi_b,
            'processor': processor,
            'absorbing_state': absorbing_state
        })
        cfg.add({'models': param['models']})

        runner.add(cfg)

    results = runner.run()

    # print results
    for result in results:
        analysis(result)

if __name__ == '__main__':
    # Local:
    # python example_tabular.py tabular_example_cfg.json

    parser = argparse.ArgumentParser(description='Distribute experiments across ec2 instances.')

    parser.add_argument('cfg', help='config file', type=str)
    args = parser.parse_args()

    with open('cfgs/{0}'.format(args.cfg), 'r') as f:
        param = json.load(f)

    main(param)
