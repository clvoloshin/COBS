import numpy as np
import argparse
import json
from copy import deepcopy

from ope.envs.gridworld import Gridworld
from ope.policies.epsilon_greedy_policy import EGreedyPolicy
from ope.policies.tabular_model import TabularPolicy

from ope.experiment_tools.experiment import ExperimentRunner, analysis
from ope.experiment_tools.config import Config
from ope.experiment_tools.factory import setup_params

def main(param):

    param = setup_params(param)
    runner = ExperimentRunner()

    for N in range(5): 
        configuration = deepcopy(param['experiment']) # Make sure to deepcopy as to never change original
        configuration['num_traj'] = 8*2**N # Increase dataset size
        
        cfg = Config(configuration)

        env = Gridworld(slippage=.2*cfg.stochastic_env)

        np.random.seed(cfg.seed)
        eval_policy = cfg.eval_policy
        base_policy = cfg.base_policy

        # to_grid and from_grid are particular to Gridworld
        # These functions are special to convert an index in a grid to an 'image'
        def to_grid(x, gridsize=[8, 8]):
            x = x.reshape(-1)
            x = x[0]
            out = np.zeros(gridsize)
            if x >= 64:
                return out
            else:
                out[x//gridsize[0], x%gridsize[1]] = 1.
            return out

        # This function takes an 'image' and returns the position in the grid
        def from_grid(x, gridsize=[8, 8]):
            if len(x.shape) == 3:
                if np.sum(x) == 0:
                    x = np.array([gridsize[0] * gridsize[1]])
                else:
                    x = np.array([np.argmax(x.reshape(-1))])
            return x

        processor = lambda x: x
        policy = env.best_policy()
        absorbing_state = processor(np.array([len(policy)]))

        pi_e = EGreedyPolicy(model=TabularPolicy(policy, absorbing=absorbing_state), processor=from_grid, prob_deviation=eval_policy, action_space_dim=env.n_actions)
        pi_b = EGreedyPolicy(model=TabularPolicy(policy, absorbing=absorbing_state), processor=from_grid, prob_deviation=base_policy, action_space_dim=env.n_actions)

        cfg.add({
            'env': env,
            'pi_e': pi_e,
            'pi_b': pi_b,
            'processor': processor,
            'absorbing_state': absorbing_state,
            'convert_from_int_to_img': to_grid,
        })
        cfg.add({'models': param['models']})

        runner.add(cfg)

    results = runner.run()

    # print results
    for result in results:
        analysis(result)

if __name__ == '__main__':
    # Local:
    # python example3.py nn_example_cfg.json

    parser = argparse.ArgumentParser(description='Distribute experiments across ec2 instances.')

    parser.add_argument('cfg', help='config file', type=str)
    args = parser.parse_args()

    with open('cfgs/{0}'.format(args.cfg), 'r') as f:
        param = json.load(f)

    main(param)
