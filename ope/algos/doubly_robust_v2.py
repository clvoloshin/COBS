import sys
import numpy as np
import pandas as pd
from functools import reduce
from ope.models.basics import BasicPolicy, SingleTrajectory
from ope.models.epsilon_greedy_policy import EGreedyPolicy
from tqdm import tqdm
import itertools

class DoublyRobust_v2(object):
    """Algorithm: Doubly Robust (DR).
    """
    def __init__(self, gamma):
        """
        Parameters
        ----------
        gamma : float
            Discount factor.
        """
        self.gamma = gamma

    def evaluate(self, info, is_wdr=False, return_Qs=False):
        """Get DR estimate from Q + IPS.

        Parameters
        ----------
        info : list
            [list of actions, list of rewards, list of base propensity, list of target propensity, list of Qhat]
        is_wdr : bool
            Use Weighted Doubly Robust?
        return_Qs : bool
            Return trajectory-wise estimate alongside full DR estimate? 
        
        Returns
        -------
        float
            DR estimate

            If return_Qs is true, also returns trajectory-wise estimate
        """
        
        (actions,
        rewards,
        base_propensity,
        target_propensities,
        estimated_q_values) = DoublyRobust_v2.transform_to_equal_length_trajectories(*info)

        num_trajectories = actions.shape[0]
        trajectory_length = actions.shape[1]

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
        importance_weights = DoublyRobust_v2.normalize_importance_weights(
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

        out = DoublyRobust_v2.calculate_step_return(
                    rewards,
                    discounts,
                    importance_weights,
                    importance_weights_one_earlier,
                    estimated_state_values,
                    estimated_q_values_for_logged_action,
                )

        if return_Qs:
            return np.array(out).sum(), np.array(out)
        else:
            return np.array(out).sum()

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
    def calculate_step_return(
        rewards,
        discounts,
        importance_weights,
        importance_weights_one_earlier,
        estimated_state_values,
        estimated_q_values,
    ):
        # Modification of Magic.
        # with j_step = T-1, this is equiavalent to DR or WDR
        trajectory_length = len(rewards[0])
        num_trajectories = len(rewards)
        j_step = int(trajectory_length - 1)

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

        # n = len(self.trajectories)
        # AM = []
        # dr_b = []
        # wdr_b = []
        # all_part_bs = []

        # print('getting Rho and Norms')
        # rho_t, norms = self.get_weights(pi_b, pi_e)
        # V = {}
        # Q = {}

        # # g = []
        # # for _ in range(1000): g.append(model.V(pi_e, 0, 0) )
        # # print('AM:', np.mean(g))
        # print('Running DR/WDR')
        # ams = []
        # for i, traj in enumerate(tqdm(self.trajectories)):
        #     if self.env is not None:
        #         ams.append([model.V(pi_e, self.env.pos_to_image(traj['x'][0:1]), 0) for model in models])
        #     else:
        #         ams.append([model.V(pi_e, traj['x'][0], 0) for model in models])

        #     # import pdb; pdb.set_trace()
        #     df = pd.DataFrame(traj)

        #     # model = models[0]
        #     # model.Q(pi_e, self.env.pos_to_image(traj['x']), traj['a'],0)
        #     # model.V(pi_e, self.env.pos_to_image(traj['x_prime']), 1)

        #     import pdb; pdb.set_trace()

        #     vals_per_model = []
        #     for model in models:
        #         if self.env is not None:
        #             vals_per_model.append(np.array([[model.Q(pi_e, self.env.pos_to_image([s]), a,t), model.V(pi_e, self.env.pos_to_image([s_prime]), t+1)] for t,(s,a,s_prime) in enumerate(np.array(df[['x','a','x_prime']])) ]))
        #         else:
        #             vals_per_model.append(np.array([[model.Q(pi_e, s, a,t), model.V(pi_e, s_prime, t+1)] for t,(s,a,s_prime) in enumerate(np.array(df[['x','a','x_prime']])) ]))

        #     qs = []
        #     vs = []
        #     for vals in vals_per_model:
        #         qs.append(vals[:,0])
        #         vs.append(vals[:,1])

        #     if self.gamma == 1:
        #         # special case -- no discounting means less calculations
        #         part_b = []
        #         for q,v in zip(qs,vs):
        #             part_b.append(rho_t[i] * (df['r'] - q + v))
        #             # df['dr_b'] = rho_t[i] * (df['r'] - q + v)
        #             # df['wdr_b'] = rho_t[i] * (df['r'] - q + v)
        #     else:
        #         raise NotImplemented

        #     part_b = np.array(part_b).T
        #     all_part_bs.append(part_b)

        #     dr_b = np.array([x.sum(axis = 0) for x in all_part_bs])
        #     wdr_b = np.vstack([(x/norms[:len(x),np.newaxis]).sum(axis = 0) for x in all_part_bs])
        #     AM = np.mean(ams, axis =0)
        #     print(AM, AM + np.mean(dr_b, axis = 0), AM + sum(wdr_b))


        # # all_part_bs = np.array(all_part_bs)

        # # dr_b = all_part_bs.sum(axis = 1)
        # # wdr_b = all_part_bs.sum(axis = 0)/norms[:,np.newaxis]
        # # AM = np.mean(ams, axis =0)
        # # return AM, AM + np.mean(dr_b, axis = 0), AM + sum(wdr_b)

        # dr_b = np.array([x.sum(axis = 0) for x in all_part_bs])
        # wdr_b = np.vstack([(x/norms[:len(x),np.newaxis]).sum(axis = 0) for x in all_part_bs])
        # AM = np.mean(ams, axis =0)
        # return AM, AM + np.mean(dr_b, axis = 0), AM + sum(wdr_b)

    def sample(self, dic):
        try:
            out = np.random.choice(list(dic), p=list(dic.values()))
        except:
            import pdb; pdb.set_trace()
        return out

    def get_weights(self, pi_b, pi_e):

        if isinstance(pi_e, BasicPolicy) or isinstance(pi_e, SingleTrajectory):
            pi_e_a_given_x = [pi_e.predict(episode['x'])[range(len(episode['a'])), np.array(episode['a']).reshape(-1)] for episode in self.trajectories]#[(pi_e(episode['x']) == episode['a']).astype(float) for episode in self.trajectories]
        elif isinstance(pi_e, EGreedyPolicy):
            pass # pi_e_a_given_x = [pi_e.predict(self.env.pos_to_image(np.array(episode['x'])))[range(len(episode['a'])), np.array(episode['a']).reshape(-1)] for episode in self.trajectories]#[(pi_e(episode['x']) == episode['a']).astype(float) for episode in self.trajectories]
        else:
            raise NotImplemented

        if isinstance(pi_b, BasicPolicy):
            pi_b_a_given_x = [pi_b.predict(episode['x'])[range(len(episode['a'])), np.array(episode['a']).reshape(-1)] for episode in self.trajectories]
        elif isinstance(pi_b, EGreedyPolicy):
            pass #pi_b_a_given_x = [pi_b.predict(self.env.pos_to_image(np.array(episode['x'])))[range(len(episode['a'])), np.array(episode['a']).reshape(-1)] for episode in self.trajectories]#[(pi_e(episode['x']) == episode['a']).astype(float) for episode in self.trajectories]
        else:
            raise NotImplemented

        if isinstance(pi_b, EGreedyPolicy):
            pi_e_a_given_x = []
            pi_b_a_given_x = []
            for episode in tqdm(self.trajectories):
                # assume model behind pi_e and pi_b is the same up to devation w prob epsilon
                acts = np.argmax(pi_e.model.predict(self.env.pos_to_image(episode['x'])), axis=1)
                pi_b_probs = np.ones((len(episode['x']), pi_b.action_space_dim)) * (pi_b.prob_deviation/pi_b.action_space_dim)
                pi_e_probs = np.ones((len(episode['x']), pi_e.action_space_dim)) * (pi_e.prob_deviation/pi_e.action_space_dim)

                pi_b_probs[range(len(acts)), acts] = 1-pi_b_probs.sum(axis=1)
                pi_e_probs[range(len(acts)), acts] = 1-pi_e_probs.sum(axis=1)

                pi_b_a_given_x.append(pi_b_probs[range(len(episode['a'])), episode['a']])
                pi_e_a_given_x.append(pi_e_probs[range(len(episode['a'])), episode['a']])


        pi_e_cumprod = [np.cumprod(x) for x in pi_e_a_given_x]
        pi_b_cumprod = [np.cumprod(x) for x in pi_b_a_given_x]
        rho_t = [pi_e_cumprod[i]/pi_b_cumprod[i] for i in range(len(pi_e_cumprod))]
        def sum_arrays(x,y):
            max_len = max(len(x), len(y))
            x = np.pad(x, (0,max_len-len(x)), mode='constant', constant_values=0)
            y = np.pad(y, (0,max_len-len(y)), mode='constant', constant_values=0)
            return x+y

        norms = reduce(lambda x,y,s_a=sum_arrays: s_a(x,y), rho_t)
        how_many_non_zero = np.sum(norms>0)

        norms[how_many_non_zero:] = 1.

        return rho_t, norms



