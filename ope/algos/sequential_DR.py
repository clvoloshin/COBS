import logging
from typing import List

import numpy as np

class SeqDoublyRobust(object):
    """Algorithm: Sequential Doubly Robust (DR).
    """
    def __init__(self, gamma):
        """
        Parameters
        ----------
        gamma : float
            Discount factor.
        """
        self.gamma = gamma

    def evaluate(self, info):
        """Get Seq-DR estimate from Q + IPS.

        Parameters
        ----------
        info : list
            [list of actions, list of rewards, list of base propensity, list of target propensity, list of Qhat]
        
        Returns
        -------
        list
            [Seq-DR estimate, normalized estimate]
        """

        (actions,
        rewards,
        base_propensity,
        target_propensities,
        estimated_q_values) = info

        num_actions = len(target_propensities[0][0])
        actions = [np.eye(num_actions)[act] for act in actions]

        base_propensity_for_logged_action = [np.sum(np.multiply(bp, acts), axis=1) for bp, acts in zip(base_propensity, actions)]
        target_propensity_for_logged_action = [np.sum(np.multiply(tp, acts), axis=1) for tp, acts in zip(target_propensities, actions)]
        estimated_q_values_for_logged_action = [np.sum(np.multiply(q, acts), axis=1) for q, acts in zip(estimated_q_values, actions)]
        estimated_state_values = [np.sum(np.multiply(p, q), axis=1) for p, q in zip(target_propensities, estimated_q_values)]


        importance_weights = [ np.array(p_target)/np.array(p_base) for p_target, p_base in zip(target_propensity_for_logged_action, base_propensity_for_logged_action)]

        doubly_robusts = []
        episode_values = []

        for episode in range(len(actions)):
            episode_value = 0.0
            doubly_robust = 0.0
            for j in range(len(actions[episode]))[::-1]:
                doubly_robust = estimated_state_values[episode][j] + importance_weights[episode][j] * (
                    rewards[episode][j]
                    + self.gamma * doubly_robust
                    - estimated_q_values_for_logged_action[episode][j]
                )
                episode_value *= self.gamma
                episode_value += rewards[episode][j]
            if episode_value > 1e-6 or episode_value < -1e-6:
                doubly_robusts.append(float(doubly_robust))
                episode_values.append(float(episode_value))

        doubly_robusts = np.array(doubly_robusts)
        episode_values = np.array(episode_values)

        denominator = np.mean(episode_values)
        if abs(denominator) < 1e-6:
            [0., 0.]

        dr_score = np.mean(doubly_robusts)
        return [float(dr_score), float(dr_score) / denominator]
