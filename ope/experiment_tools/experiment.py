import json
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.signal as signal
import os
from skimage.transform import rescale, resize, downscale_local_mean
import json
from collections import OrderedDict, Counter
import time
import argparse
import boto3
import glob
import numpy as np
import sys
from pdb import set_trace as b
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize

from ope.algos.doubly_robust_v2 import DoublyRobust_v2 as DR
from ope.algos.fqe import FittedQEvaluation
from ope.algos.magic import MAGIC
from ope.algos.average_model import AverageModel as AM
from ope.algos.sequential_DR import SeqDoublyRobust as SeqDR
from ope.algos.dm_regression import DirectMethodRegression as DM
from ope.algos.traditional_is import TraditionalIS as IS
from ope.algos.infinite_horizon import InfiniteHorizonOPE as IH
from ope.algos.dm_regression import DirectMethodRegression
from ope.algos.more_robust_doubly_robust import MRDR
from ope.algos.retrace_lambda import Retrace
from ope.algos.approximate_model import ApproxModel

from ope.policies.basics import BasicPolicy
from ope.policies.epsilon_greedy_policy import EGreedyPolicy
from ope.policies.max_likelihood import MaxLikelihoodModel
from ope.policies.Q_wrapper import QWrapper
from ope.policies.tabular_model import TabularPolicy

from ope.utls.get_Qs import getQs
from ope.utls.rollout import rollout

def analysis(dic):

    divergence = -1
    if 'KLDivergence' in dic:
        divergence = dic['KLDivergence']
        del dic['KLDivergence']

    longest = max([len(key) for key,_ in dic.items()])
    sorted_keys = np.array([[key,val[1]] for key,val in dic.items()])
    sorted_keys = sorted_keys[np.argsort(sorted_keys[:,1].astype(float))]

    # sorted_keys = sorted_keys[sorted(sorted_ke)]
    print ("Results:  \n")
    for key, value in dic.items():
        label = ' '*(longest-len(key)) + key
        print("{}: {:10.4f}. Error: {:10.4f}".format(label, *value))
    print('\n')
    print ("Ordered Results:  \n")
    for key in sorted_keys[:,0]:
        value = dic[key]
        label = ' '*(longest-len(key)) + key
        print("{}: {:10.4f}. Error: {:10.4f}".format(label, *value))

    dic['KLDivergence'] = divergence
    return dic

class Result(object):
    def __init__(self, cfg, result):
        self.cfg = cfg
        self.result = result

class ExperimentRunner(object):
    def __init__(self):
        self.results = []
        self.cfgs = []

    def add(self, cfg):
        self.cfgs.append(cfg)

    def run(self):
        results = []

        for cfg in self.cfgs:
            if cfg.modeltype == 'tabular':
                result = self.single_run(cfg)
            elif cfg.modeltype == 'linear':
                result = None
            else:
                result = self.single_run(cfg)

            results.append(result)

        return results

    def get_rollout(self, cfg, eval_data=False, N_overwrite = None):
        env = cfg.env
        pi_e = cfg.pi_e
        pi_b = cfg.pi_b
        processor = cfg.processor
        absorbing_state = cfg.absorbing_state
        T = cfg.horizon
        frameskip = cfg.frameskip if cfg.frameskip is not None else 1
        frameheight = cfg.frameheight if cfg.frameheight is not None else 1
        use_only_last_reward = cfg.use_only_last_reward if cfg.use_only_last_reward is not None else False

        if eval_data:
            data = rollout(env, pi_e, processor, absorbing_state, N=max(10000, cfg.num_traj) if N_overwrite is None else N_overwrite, T=T, frameskip=frameskip, frameheight=frameheight, path=None, filename='tmp', use_only_last_reward=use_only_last_reward)
        else:
            data = rollout(env, pi_b, processor, absorbing_state, pi_e = pi_e, N=cfg.num_traj, T=T, frameskip=frameskip, frameheight=frameheight, path=None, filename='tmp',use_only_last_reward=use_only_last_reward)

        return data

    def single_run(self, cfg):
        env = cfg.env
        pi_e = cfg.pi_e
        pi_b = cfg.pi_b
        processor = cfg.processor
        absorbing_state = cfg.absorbing_state
        T = cfg.horizon
        gamma = cfg.gamma
        models = cfg.models
        frameskip = cfg.frameskip
        frameheight = cfg.frameheight
        modeltype = cfg.modeltype
        Qmodel = cfg.Qmodel

        dic = {}

        eval_data = rollout(env, pi_e, processor, absorbing_state, N=max(10000, cfg.num_traj), T=T, frameskip=frameskip, frameheight=frameheight, path=None, filename='tmp',)
        behavior_data = rollout(env, pi_b, processor, absorbing_state, pi_e = pi_e, N=cfg.num_traj, T=T, frameskip=frameskip, frameheight=frameheight, path=None, filename='tmp',)

        if cfg.convert_from_int_to_img is not None:
            traj = []
            for trajectory in behavior_data.trajectories:
                frames = []
                for frame in trajectory['frames']:
                    frames.append(cfg.convert_from_int_to_img(np.array(frame)))
                traj.append(frames)
            for i,frames in enumerate(traj):
                behavior_data.trajectories[i]['frames'] = frames

        if cfg.to_regress_pi_b:
            behavior_data.estimate_propensity()

        true = eval_data.value_of_data(gamma, False)
        dic.update({'ON POLICY': [float(true), 0]})
        print('V(pi_b): ',behavior_data.value_of_data(gamma, False), 'V(pi_b) Normalized: ',behavior_data.value_of_data(gamma, True))
        print('V(pi_e): ',eval_data.value_of_data(gamma, False), 'V(pi_e) Normalized: ',eval_data.value_of_data(gamma, True))

        for model in cfg.models:
            if 'FQE' == model:
                FQE = FittedQEvaluation()
                FQE.fit(behavior_data, pi_e, cfg, cfg.models[model]['model'])
                FQE_Qs = FQE.get_Qs_for_data(behavior_data, cfg)
                out = self.estimate(FQE_Qs, behavior_data, gamma, model, true)
                dic.update(out)
            elif 'Retrace' == model:
                retrace = Retrace(model, cfg.models[model]['lamb'])
                retrace.fit(behavior_data, pi_e, cfg, cfg.models[model]['model'])
                retrace_Qs = retrace.get_Qs_for_data(behavior_data, cfg)
                out = self.estimate(retrace_Qs, behavior_data, gamma, model, true)
                dic.update(out)
            elif 'Tree-Backup' == model:
                tree = Retrace(model, cfg.models[model]['lamb'])
                tree.fit(behavior_data, pi_e, cfg, cfg.models[model]['model'])
                tree_Qs = tree.get_Qs_for_data(behavior_data, cfg)
                out = self.estimate(tree_Qs, behavior_data, gamma, model, true)
                dic.update(out)
            elif 'Q^pi(lambda)' == model:
                q_lambda = Retrace(model, cfg.models[model]['lamb'])
                q_lambda.fit(behavior_data, pi_e, cfg, cfg.models[model]['model'])
                q_lambda_Qs = q_lambda.get_Qs_for_data(behavior_data, cfg)
                out = self.estimate(q_lambda_Qs, behavior_data, gamma, model, true)
                dic.update(out)
            elif 'Q-Reg' == model:
                q_reg = DirectMethodRegression()
                q_reg.fit(behavior_data, pi_e, cfg, cfg.models[model]['model'])
                q_reg_Qs = q_reg.get_Qs_for_data(behavior_data, cfg)
                out = self.estimate(q_reg_Qs, behavior_data, gamma, model, true)
                dic.update(out)
            elif 'MRDR' == model:
                mrdr = MRDR()
                mrdr.fit(behavior_data, pi_e, cfg, cfg.models[model]['model'])
                mrdr_Qs = mrdr.get_Qs_for_data(behavior_data, cfg)
                out = self.estimate(mrdr_Qs, behavior_data, gamma, model, true)
                dic.update(out)
            elif 'IH' == model:
                ih = IH()
                ih.fit(behavior_data, pi_e, cfg, cfg.models[model]['model'])
                inf_hor_output = ih.evaluate(behavior_data, cfg)
                dic.update({'IH': [inf_hor_output, (inf_hor_output - true )**2]})
            elif 'MBased' == model:
                mbased = ApproxModel(cfg, behavior_data.n_actions)
                mbased.fit(behavior_data, pi_e, cfg, cfg.models[model]['model'])
                mbased.get_Qs_for_data(pi_e, behavior_data, cfg)
            elif 'IS' == model:
                out = self.estimate([], behavior_data, gamma, 'IS', true, True)
                dic.update(out)
            else:
                print(model, ' is not a valid method')
            
        # for model in models:
        #     if (model == 'MBased_Approx') or (model == 'MBased_MLE'):
        #         if model == 'MBased_MLE':
        #             print('*'*20)
        #             print('MLE estimator not implemented for continuous state space. Using MBased_Approx instead')
        #             print('*'*20)
        #         MBased_max_trajectory_length = 25
        #         batchsize = 32
        #         mbased_num_epochs = 100
        #         MDPModel = ApproxModel(gamma, None, MBased_max_trajectory_length, frameskip, frameheight, processor, action_space_dim=env.n_actions)
        #         mdpmodel = MDPModel.run(env, behavior_data, mbased_num_epochs, batchsize, Qmodel)

        #         Qs_model_based = get_Qs.get(mdpmodel)
        #         out = self.estimate(Qs_model_based, behavior_data, gamma,'MBased_Approx', true)
        #         dic.update(out)


        result = analysis(dic)
        self.results.append(Result(cfg, result))
        return result

    def estimate(self, Qs, data, gamma, name, true, IS_eval=False):
        dic = {}
        dr = DR(gamma)
        mag = MAGIC(gamma)
        am = AM(gamma)
        sdr = SeqDR(gamma)
        imp_samp = IS(gamma)
        num_j_steps = 25

        info = [data.actions(),
                data.rewards(),
                data.base_propensity(),
                data.target_propensity(),
                Qs
                ]

        if IS_eval:
            IS_eval = imp_samp.evaluate(info)
            dic['NAIVE']     = [float(IS_eval[0]), float( (IS_eval[0] - true )**2)]
            dic['IS']        = [float(IS_eval[1]), float( (IS_eval[1] - true )**2)]
            dic['STEP IS']   = [float(IS_eval[2]), float( (IS_eval[2] - true )**2)]
            dic['WIS']       = [float(IS_eval[3]), float( (IS_eval[3] - true )**2)]
            dic['STEP WIS']  = [float(IS_eval[4]), float( (IS_eval[4] - true )**2)]
        else:
            dr_evaluation = dr.evaluate(info)
            wdr_evaluation = dr.evaluate(info, True)
            magic_evaluation = mag.evaluate(info, num_j_steps, True)
            AM_evaluation = am.evaluate(info)
            SDR_evaluation = sdr.evaluate(info)
            dic['AM {0}'.format(name)] = [AM_evaluation, (AM_evaluation - true)**2]
            dic['DR {0}'.format(name)] = [dr_evaluation, (dr_evaluation - true)**2]
            dic['WDR {0}'.format(name)] = [wdr_evaluation, (wdr_evaluation - true)**2]
            dic['MAGIC {0}'.format(name)] = [magic_evaluation[0], (magic_evaluation[0] - true )**2]
            dic['SDR {0}'.format(name)] = [SDR_evaluation[0], (SDR_evaluation[0] - true )**2]

        # return dr_evaluation, wdr_evaluation, magic_evaluation, AM_evaluation, SDR_evaluation
        return dic
