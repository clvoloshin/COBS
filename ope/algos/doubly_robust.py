import sys
import numpy as np
import pandas as pd
from functools import reduce
from ope.models.basics import BasicPolicy, SingleTrajectory
from ope.models.epsilon_greedy_policy import EGreedyPolicy
from tqdm import tqdm

class DoublyRobust(object):
    def __init__(self, trajectories, gamma, env=None):
        self.trajectories = trajectories
        self.gamma = gamma
        self.env = env

    def run(self, pi_b, pi_e, models):
        n = len(self.trajectories)
        AM = []
        dr_b = []
        wdr_b = []
        all_part_bs = []

        print('getting Rho and Norms')
        rho_t, norms = self.get_weights(pi_b, pi_e)
        V = {}
        Q = {}

        # g = []
        # for _ in range(1000): g.append(model.V(pi_e, 0, 0) )
        # print('AM:', np.mean(g))
        print('Running DR/WDR')
        ams = []
        for i, traj in enumerate(tqdm(self.trajectories)):
            if self.env is not None:
                ams.append([model.V(pi_e, self.env.pos_to_image(traj['x'][0:1]), 0) for model in models])
            else:
                ams.append([model.V(pi_e, traj['x'][0], 0) for model in models])

            # import pdb; pdb.set_trace()
            df = pd.DataFrame(traj)

            # model = models[0]
            # model.Q(pi_e, self.env.pos_to_image(traj['x']), traj['a'],0)
            # model.V(pi_e, self.env.pos_to_image(traj['x_prime']), 1)

            import pdb; pdb.set_trace()

            vals_per_model = []
            for model in models:
                if self.env is not None:
                    vals_per_model.append(np.array([[model.Q(pi_e, self.env.pos_to_image([s]), a,t), model.V(pi_e, self.env.pos_to_image([s_prime]), t+1)] for t,(s,a,s_prime) in enumerate(np.array(df[['x','a','x_prime']])) ]))
                else:
                    vals_per_model.append(np.array([[model.Q(pi_e, s, a,t), model.V(pi_e, s_prime, t+1)] for t,(s,a,s_prime) in enumerate(np.array(df[['x','a','x_prime']])) ]))

            qs = []
            vs = []
            for vals in vals_per_model:
                qs.append(vals[:,0])
                vs.append(vals[:,1])

            if self.gamma == 1:
                # special case -- no discounting means less calculations
                part_b = []
                for q,v in zip(qs,vs):
                    part_b.append(rho_t[i] * (df['r'] - q + v))
                    # df['dr_b'] = rho_t[i] * (df['r'] - q + v)
                    # df['wdr_b'] = rho_t[i] * (df['r'] - q + v)
            else:
                raise NotImplemented

            part_b = np.array(part_b).T
            all_part_bs.append(part_b)

            dr_b = np.array([x.sum(axis = 0) for x in all_part_bs])
            wdr_b = np.vstack([(x/norms[:len(x),np.newaxis]).sum(axis = 0) for x in all_part_bs])
            AM = np.mean(ams, axis =0)
            print(AM, AM + np.mean(dr_b, axis = 0), AM + sum(wdr_b))


        # all_part_bs = np.array(all_part_bs)

        # dr_b = all_part_bs.sum(axis = 1)
        # wdr_b = all_part_bs.sum(axis = 0)/norms[:,np.newaxis]
        # AM = np.mean(ams, axis =0)
        # return AM, AM + np.mean(dr_b, axis = 0), AM + sum(wdr_b)

        dr_b = np.array([x.sum(axis = 0) for x in all_part_bs])
        wdr_b = np.vstack([(x/norms[:len(x),np.newaxis]).sum(axis = 0) for x in all_part_bs])
        AM = np.mean(ams, axis =0)
        return AM, AM + np.mean(dr_b, axis = 0), AM + sum(wdr_b)

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



