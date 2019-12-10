
import numpy as np
import itertools

class AverageModel(object):
    def __init__(self, gamma):
        self.gamma = gamma

    def evaluate(self, info):
        (actions,
        rewards,
        base_propensity,
        target_propensities,
        estimated_q_values) = AverageModel.transform_to_equal_length_trajectories(*info)

        num_trajectories = actions.shape[0]
        trajectory_length = actions.shape[1]

        estimated_state_values = np.sum(
            np.multiply(target_propensities, estimated_q_values), axis=2
        )

        return np.mean([V[0] for V in estimated_state_values])

    @staticmethod
    def transform_to_equal_length_trajectories(
        actions,
        rewards,
        logged_propensities,
        target_propensities,
        estimated_q_values,
    ):
        """
        Take in samples (action, rewards, propensities, etc.) and output lists
        of equal-length trajectories (episodes) accoriding to terminals.
        As the raw trajectories are of various lengths, the shorter ones are
        filled with zeros(ones) at the end.
        """
        num_actions = len(target_propensities[0][0])
        
        def to_equal_length(x, fill_value):
            x_equal_length = np.array(
                list(itertools.zip_longest(*x, fillvalue=fill_value))
            ).swapaxes(0, 1)
            return x_equal_length

        action_trajectories = to_equal_length(
            [np.eye(num_actions)[act] for act in actions], np.zeros([num_actions])
        )
        reward_trajectories = to_equal_length(rewards, 0)
        logged_propensity_trajectories = to_equal_length(
            logged_propensities, np.zeros([num_actions])
        )
        target_propensity_trajectories = to_equal_length(
            target_propensities, np.zeros([num_actions])
        )

        # Hack for now. Delete.
        estimated_q_values = [[np.hstack(y).tolist() for y in x] for x in estimated_q_values]

        Q_value_trajectories = to_equal_length(
            estimated_q_values, np.zeros([num_actions])
        )

        return (
            action_trajectories,
            reward_trajectories,
            logged_propensity_trajectories,
            target_propensity_trajectories,
            Q_value_trajectories,
        )

        