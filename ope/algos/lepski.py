
import itertools

import numpy as np
import scipy as sp


class LEPSKI(object):

    def __init__(self, gamma):
        self.gamma = gamma

    def evaluate(self, info, num_j_steps, true, is_wdr):

        (actions,
         rewards,
         base_propensity,
         target_propensities,
         estimated_q_values) = LEPSKI.transform_to_equal_length_trajectories(*info)

        num_trajectories = actions.shape[0]
        trajectory_length = actions.shape[1]

        num_j_steps = trajectory_length + 1

        j_steps = [i * 1 for i in range(-1, trajectory_length)]

        base_propensity_for_logged_action = np.sum(
            np.multiply(base_propensity, actions), axis=2
        )
        target_propensity_for_logged_action = np.sum(
            np.multiply(target_propensities, actions), axis=2
        )
        estimated_q_values_for_logged_action = np.sum(
            np.multiply(estimated_q_values, actions), axis=2
        )
        estimated_state_values = np.sum(
            np.multiply(target_propensities, estimated_q_values), axis=2
        )

        importance_weights = target_propensity_for_logged_action / \
            base_propensity_for_logged_action
        importance_weights[np.isnan(importance_weights)] = 0.
        importance_weights = np.cumprod(importance_weights, axis=1)
        importance_weights = LEPSKI.normalize_importance_weights(
            importance_weights, is_wdr
        )

        importance_weights_one_earlier = (
            np.ones([num_trajectories, 1]) * 1.0 / num_trajectories
        )
        importance_weights_one_earlier = np.hstack(
            [importance_weights_one_earlier, importance_weights[:, :-1]]
        )

        discounts = np.logspace(
            start=0, stop=trajectory_length - 1, num=trajectory_length, base=self.gamma
        )

        j_step_return_trajectories = []
        for j_step in j_steps:
            j_step_return_trajectories.append(
                LEPSKI.calculate_step_return(
                    rewards,
                    discounts,
                    importance_weights,
                    importance_weights_one_earlier,
                    estimated_state_values,
                    estimated_q_values_for_logged_action,
                    j_step,
                )
            )
        j_step_return_trajectories = np.array(j_step_return_trajectories)

        j_step_returns = np.sum(
            j_step_return_trajectories, axis=1)  # sum over n

        stepwise_mses = np.square(j_step_returns - true)
        optimal_idx = np.argmin(stepwise_mses)
        optimal_est = j_step_returns[optimal_idx]
        print(f"[Lepski]: optimal idx = {optimal_idx}")
        print(f"[Lepski]: optimal est = {optimal_est}")

        if len(j_step_returns) == 1:
            lepski = j_step_returns[0]
        else:
            lepski = self.compute_lepski_point_estimate(
                j_steps,
                num_j_steps,
                j_step_returns,
                j_step_return_trajectories,
            )

        episode_values = np.sum(np.multiply(rewards, discounts), axis=1)
        denominator = np.nanmean(episode_values)
        if abs(denominator) < 1e-6:
            return [0]*1

        return [lepski, optimal_est]

    def compute_lepski_point_estimate(
        self,
        j_steps,
        num_j_steps,
        j_step_returns,
        j_step_return_trajectories,
    ):
        constant_param = 2
        means = []
        intervals = []
        num_trajectories = j_step_return_trajectories.shape[1]  # check

        var_list = [0] * num_j_steps

        for j in range(num_j_steps):
            var = np.var(j_step_return_trajectories[j, :]) * num_trajectories
            var_list[j] = var

        for j in range(num_j_steps):
            mean = j_step_returns[j]
            means.append(mean)

            if j < num_j_steps - 1:
                new_var = np.min(var_list[j:])
            else:
                new_var = var_list[j]

            intervals.append((mean - constant_param * np.sqrt(new_var),
                              mean + constant_param * np.sqrt(new_var)))
            print(
                f"[Lepski]: mean = {mean}, low = {intervals[-1][0]}, high = {intervals[-1][1]}", flush=True)

        index = num_j_steps - 1
        curr = [intervals[-1][0], intervals[-1][1]]
        for i in range(num_j_steps-1, -1, -1):
            if intervals[i][0] > curr[1] or intervals[i][1] < curr[0]:
                print(f"[Lepski]: break point = {i}")
                break
            else:
                curr[0] = max(curr[0], intervals[i][0])
                curr[1] = min(curr[1], intervals[i][1])
                index = i
            print(
                f"[Lepski]: current low = {curr[0]}, current high = {curr[1]}")
        print(f"[Lepski]: returning index = {index}", flush=True)

        return j_step_returns[index]

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
            [np.eye(num_actions)[act]
             for act in actions], np.zeros([num_actions])
        )
        reward_trajectories = to_equal_length(rewards, 0)
        logged_propensity_trajectories = to_equal_length(
            logged_propensities, np.zeros([num_actions])
        )
        target_propensity_trajectories = to_equal_length(
            target_propensities, np.zeros([num_actions])
        )

        # Hack for now. Delete.
        estimated_q_values = [[np.hstack(y).tolist()
                               for y in x] for x in estimated_q_values]

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

    @staticmethod
    def normalize_importance_weights(
        importance_weights, is_wdr
    ):
        if is_wdr:
            sum_importance_weights = np.sum(importance_weights, axis=0)
            where_zeros = np.where(sum_importance_weights == 0.0)[0]
            sum_importance_weights[where_zeros] = len(importance_weights)
            importance_weights[:, where_zeros] = 1.0
            importance_weights /= sum_importance_weights
            return importance_weights
        else:
            importance_weights /= importance_weights.shape[0]
            return importance_weights

    @staticmethod
    def calculate_step_return(
        rewards,
        discounts,
        importance_weights,
        importance_weights_one_earlier,
        estimated_state_values,
        estimated_q_values,
        j_step,
    ):
        trajectory_length = len(rewards[0])
        num_trajectories = len(rewards)
        j_step = int(min(j_step, trajectory_length - 1))

        weighted_discounts = np.multiply(discounts, importance_weights)
        weighted_discounts_one_earlier = np.multiply(
            discounts, importance_weights_one_earlier
        )

        importance_sampled_cumulative_reward = np.sum(
            np.multiply(
                weighted_discounts[:, : j_step + 1], rewards[:, : j_step + 1]),
            axis=1,
        )

        if j_step < trajectory_length - 1:
            direct_method_value = (
                weighted_discounts_one_earlier[:, j_step + 1]
                * estimated_state_values[:, j_step + 1]
            )
        else:
            direct_method_value = np.zeros([num_trajectories])

        control_variate = np.sum(
            np.multiply(
                weighted_discounts[:, : j_step +
                                   1], estimated_q_values[:, : j_step + 1]
            )
            - np.multiply(
                weighted_discounts_one_earlier[:, : j_step + 1],
                estimated_state_values[:, : j_step + 1],
            ),
            axis=1,
        )

        j_step_return = (
            importance_sampled_cumulative_reward + direct_method_value - control_variate
        )

        return j_step_return


def mse_loss(x, error):
    return np.dot(np.dot(x, error), x.T)
