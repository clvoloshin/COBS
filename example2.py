import numpy as np

from ope.envs.gridworld import Gridworld
from ope.models.epsilon_greedy_policy import EGreedyPolicy
from ope.models.tabular_model import TabularPolicy

from ope.experiment_tools.experiment import ExperimentRunner, analysis
from ope.experiment_tools.config import Config

def main():

    runner = ExperimentRunner()

    for N in range(5):
        configuration = {
            "gamma": 0.98,
            "horizon": 5,
            "base_policy": .8,
            "eval_policy": .2,
            "stochastic_env": True,
            "stochastic_rewards": False,
            "sparse_rewards": False,
            "num_traj": 8*2**N,
            "is_pomdp": False,
            "pomdp_horizon": 2,
            "seed": 1000,
            "experiment_number": 0,
            "access": 0,
            "secret": 0,
            "to_regress_pi_b": False,
            "frameskip": 1,
            "frameheight": 1,
            "modeltype": 'conv',
            "Qmodel": 'conv1',
        }


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
        cfg.add({'models': 'all'})

        runner.add(cfg)

    results = runner.run()

    # print results
    for result in results:
        analysis(result)

if __name__ == '__main__':
    # Local:
    # python example2.py
    main()
