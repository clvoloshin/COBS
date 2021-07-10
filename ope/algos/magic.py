
import itertools

import numpy as np
import scipy as sp


class MAGIC(object):
    """Algorithm: MAGIC.
    """
    NUM_SUBSETS_FOR_CB_ESTIMATES = 25
    CONFIDENCE_INTERVAL = 0.9
    NUM_BOOTSTRAP_SAMPLES = 50
    BOOTSTRAP_SAMPLE_PCT = 0.5

    def __init__(self, gamma):
        """
        Parameters
        ----------
        gamma : float
            Discount factor.
        """
        self.gamma = gamma

    def evaluate(self, info, num_j_steps, is_wdr, return_Qs = False):
        """Get MAGIC estimate from Q + IPS.

        Parameters
        ----------
        info : list
            [list of actions, list of rewards, list of base propensity, list of target propensity, list of Qhat]
        num_j_steps : int
            Parameter to MAGIC algorithm
        is_wdr : bool
            Use Weighted Doubly Robust?
        return_Qs : bool
            Return trajectory-wise estimate alongside full DR estimate?
            Default: False
        
        Returns
        -------
        list
            [MAGIC estimate, normalized MAGIC, std error, normalized std error]

            If return_Qs is true, also returns trajectory-wise estimate
        """

        (actions,
        rewards,
        base_propensity,
        target_propensities,
        estimated_q_values) = MAGIC.transform_to_equal_length_trajectories(*info)

        num_trajectories = actions.shape[0]
        trajectory_length = actions.shape[1]

        j_steps = [float("inf")]

        if num_j_steps > 1:
            j_steps.append(-1)
        if num_j_steps > 2:
            interval = trajectory_length // (num_j_steps - 1)
            j_steps.extend([i * interval for i in range(1, num_j_steps - 1)])

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

        importance_weights = target_propensity_for_logged_action / base_propensity_for_logged_action
        importance_weights[np.isnan(importance_weights)] = 0.
        importance_weights = np.cumprod(importance_weights, axis=1)
        importance_weights = MAGIC.normalize_importance_weights(
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
                MAGIC.calculate_step_return(
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

        j_step_returns = np.sum(j_step_return_trajectories, axis=1)

        if len(j_step_returns) == 1:
            weighted_doubly_robust = j_step_returns[0]
            weighted_doubly_robust_std_error = 0.0
        else:
            # break trajectories into several subsets to estimate confidence bounds
            infinite_step_returns = []
            num_subsets = int(
                min(
                    num_trajectories / 2,
                    MAGIC.NUM_SUBSETS_FOR_CB_ESTIMATES,
                )
            )
            interval = num_trajectories / num_subsets
            for i in range(num_subsets):
                trajectory_subset = np.arange(
                    int(i * interval), int((i + 1) * interval)
                )
                importance_weights = (
                    target_propensity_for_logged_action[trajectory_subset]
                    / base_propensity_for_logged_action[trajectory_subset]
                )
                importance_weights[np.isnan(importance_weights)] = 0.
                importance_weights = np.cumprod(importance_weights, axis=1)
                importance_weights = MAGIC.normalize_importance_weights(
                    importance_weights, is_wdr
                )
                importance_weights_one_earlier = (
                    np.ones([len(trajectory_subset), 1]) * 1.0 / len(trajectory_subset)
                )
                importance_weights_one_earlier = np.hstack(
                    [importance_weights_one_earlier, importance_weights[:, :-1]]
                )
                infinite_step_return = np.sum(
                    MAGIC.calculate_step_return(
                        rewards[trajectory_subset],
                        discounts,
                        importance_weights,
                        importance_weights_one_earlier,
                        estimated_state_values[trajectory_subset],
                        estimated_q_values_for_logged_action[trajectory_subset],
                        float("inf"),
                    )
                )
                infinite_step_returns.append(infinite_step_return)

            # Compute weighted_doubly_robust mean point estimate using all data
            weighted_doubly_robust, xs = self.compute_weighted_doubly_robust_point_estimate(
                j_steps,
                num_j_steps,
                j_step_returns,
                infinite_step_returns,
                j_step_return_trajectories,
            )

            # Use bootstrapping to compute weighted_doubly_robust standard error
            bootstrapped_means = []
            sample_size = int(
                MAGIC.BOOTSTRAP_SAMPLE_PCT
                * num_subsets
            )
            for _ in range(
                MAGIC.NUM_BOOTSTRAP_SAMPLES
            ):
                random_idxs = np.random.choice(num_j_steps, sample_size, replace=False)
                random_idxs.sort()
                wdr_estimate = self.compute_weighted_doubly_robust_point_estimate(
                    j_steps=[j_steps[i] for i in random_idxs],
                    num_j_steps=sample_size,
                    j_step_returns=j_step_returns[random_idxs],
                    infinite_step_returns=infinite_step_returns,
                    j_step_return_trajectories=j_step_return_trajectories[random_idxs],
                )
                bootstrapped_means.append(wdr_estimate)
            weighted_doubly_robust_std_error = np.std(bootstrapped_means)

        episode_values = np.sum(np.multiply(rewards, discounts), axis=1)
        denominator = np.nanmean(episode_values)
        if abs(denominator) < 1e-6:
            return [0]*4

        # print (weighted_doubly_robust,
        #         weighted_doubly_robust / denominator,
        #         weighted_doubly_robust_std_error,
        #         weighted_doubly_robust_std_error / denominator)

        if return_Qs:
            return [weighted_doubly_robust,
                    weighted_doubly_robust / denominator,
                    weighted_doubly_robust_std_error,
                    weighted_doubly_robust_std_error / denominator], np.dot(xs, j_step_return_trajectories)
        else:
            return [weighted_doubly_robust,
                    weighted_doubly_robust / denominator,
                    weighted_doubly_robust_std_error,
                    weighted_doubly_robust_std_error / denominator]

    def compute_weighted_doubly_robust_point_estimate(
        self,
        j_steps,
        num_j_steps,
        j_step_returns,
        infinite_step_returns,
        j_step_return_trajectories,
    ):
        low_bound, high_bound = MAGIC.confidence_bounds(
            infinite_step_returns,
            MAGIC.CONFIDENCE_INTERVAL,
        )
        # decompose error into bias + variance
        j_step_bias = np.zeros([num_j_steps])
        where_lower = np.where(j_step_returns < low_bound)[0]
        j_step_bias[where_lower] = low_bound - j_step_returns[where_lower]
        where_higher = np.where(j_step_returns > high_bound)[0]
        j_step_bias[where_higher] = j_step_returns[where_higher] - high_bound

        covariance = np.cov(j_step_return_trajectories)
        error = covariance + j_step_bias.T * j_step_bias

        # minimize mse error
        constraint = {"type": "eq", "fun": lambda x: np.sum(x) - 1.0}

        x = np.zeros([len(j_steps)])
        res = sp.optimize.minimize(
            mse_loss,
            x,
            args=error,
            constraints=constraint,
            bounds=[(0, 1) for _ in range(x.shape[0])],
        )
        x = np.array(res.x)
        return float(np.dot(x, j_step_returns)), x

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
            np.multiply(weighted_discounts[:, : j_step + 1], rewards[:, : j_step + 1]),
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
                weighted_discounts[:, : j_step + 1], estimated_q_values[:, : j_step + 1]
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

    @staticmethod
    def confidence_bounds(x, confidence):
        n = len(x)
        m, se = np.mean(x), sp.stats.sem(x)
        h = se * sp.stats.t._ppf((1 + confidence) / 2.0, n - 1)
        return m - h, m + h


def mse_loss(x, error):
    return np.dot(np.dot(x, error), x.T)
