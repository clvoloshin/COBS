import numpy as np

from ope.envs.graph import Graph
from ope.models.basics import BasicPolicy

from ope.experiments.experiment import ExperimentRunner, analysis
from ope.experiments.config import Config


def main():

    runner = ExperimentRunner()

    # run 5 experiments
    for N in range(5):

        # basic configuration with varying number of trajectories
        configuration = {
            "gamma": 0.98,
            "horizon": 4,
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
            "modeltype": "tabular",
            "to_regress_pi_b": False,
        }

        # store them
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

        # Decide which OPE methods to run.
        # Currently only all is available
        cfg.add({'models': 'all'})

        # Add the configuration
        runner.add(cfg)

    # Run the configurations
    results = runner.run()

    # print results
    for result in results:
        analysis(result)


if __name__ == '__main__':
    # Local:
    # python example.py
    main()
