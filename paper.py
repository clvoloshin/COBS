import json
import argparse
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
import scipy.signal as signal
import os
from skimage.transform import rescale, resize, downscale_local_mean
import json
from collections import OrderedDict, Counter
import tensorflow as tf
from keras.models import load_model, model_from_json
from keras import backend as K
import time
import argparse
import boto3
import glob
import numpy as np
import sys
import pdb;
from pdb import set_trace as b
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

from ope.models.approximate_model import ApproxModel
from ope.models.basics import BasicPolicy
from ope.models.epsilon_greedy_policy import EGreedyPolicy
from ope.models.max_likelihood import MaxLikelihoodModel
from ope.models.Q_wrapper import QWrapper
from ope.models.tabular_model import TabularPolicy

from ope.utls.get_Qs import getQs
from ope.utls.rollout import rollout

'''
This is the script used for the paper. It is not really meant to be modified,
rather, should serve as a tool for replication.

See function at the bottom for details about how to run locally.
'''

def estimate(Qs, data, gamma, name, true, IS_eval=False):
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

def gridworld(param, models, debug=False):

    from ope.envs.gridworld import Gridworld
    print(param)
    stochastic_env = param['stochastic_env']
    env = Gridworld(slippage=float(.2 * stochastic_env))

    # policy = {0: 1, 1: 2, 2: 2, 3: 2, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 2, 14: 1, 15: 1, 16: 1, 17: 1, 18: 1, 19: 1, 20: 1, 21: 2, 22: 1, 23: 1, 24: 1, 25: 1, 26: 1, 27: 1, 28: 1, 29: 1, 30: 1, 31: 1, 32: 1, 33: 1, 34: 1, 35: 2, 36: 2, 37: 2, 38: 1, 39: 1, 40: 1, 41: 2, 42: 2, 43: 1, 44: 2, 45: 1, 46: 1, 47: 2, 48: 1, 49: 1, 50: 1, 51: 1, 52: 2, 53: 2, 54: 2, 55: 2, 57: 2, 58: 2, 59: 2, 60: 2, 61: 2, 62: 2, 63: 2}
    # policy = {(key//8, key%8):val for key,val in policy.items()}
    # policy = {(7-key[1], key[0]):val for key,val in policy.items()}
    # policy = {(8*key[0] + key[1]):val for key,val in policy.items()}
    policy = env.best_policy()

    np.random.seed(param['seed'])
    eval_policy = param['eval_policy']/100
    base_policy = param['base_policy']/100
    to_regress_pi_b = param['to_regress_pi_b']
    gamma = param['gamma']
    assert 0 <= gamma < 1, 'This assumes discounted case. Please make gamma < 1'
    T = param['horizon']
    processor = lambda x: x
    absorbing_state = processor(np.array([len(policy)]))
    dic = OrderedDict()

    # assert eval_policy in range(5), 'Eval: Can only choose from 5 policies'
    # assert base_policy in range(5), 'Base: Can only choose from 5 policies'
    pi_e = EGreedyPolicy(model=TabularPolicy(policy, absorbing=absorbing_state), prob_deviation=eval_policy, action_space_dim=env.n_actions)
    pi_b = EGreedyPolicy(model=TabularPolicy(policy, absorbing=absorbing_state), prob_deviation=base_policy, action_space_dim=env.n_actions)

    eval_data = rollout(env, pi_e, processor, absorbing_state, N=1024, T=T, frameskip=1, frameheight=1, path=None, filename='tmp',)
    behavior_data = rollout(env, pi_b, processor, absorbing_state, pi_e = pi_e, N=param['num_traj'], T=T, frameskip=1, frameheight=1, path=None, filename='tmp',)

    if to_regress_pi_b:
        behavior_data.estimate_propensity()

    eval_array = np.zeros(env.n_dim-1)
    counter = Counter(eval_data.states().reshape(-1))
    for key, val in counter.items():
        if key == env.terminal_state+1: continue #abs state
        eval_array[key] = val

    base_array = np.zeros(env.n_dim-1)
    counter = Counter(behavior_data.states().reshape(-1))
    for key, val in counter.items():
        if key == env.terminal_state+1: continue #abs state
        base_array[key] = val

    base_density = base_array / sum(base_array)
    base_density[base_density == 0.0] = +1e-8

    eval_density = eval_array / sum(eval_array)
    eval_density[eval_density == 0.0] = +1e-8

    divergence = np.sum(eval_density * np.log(eval_density / base_density))
    dic['KLDivergence'] = divergence

    true = eval_data.value_of_data(gamma, False)
    dic.update({'ON POLICY': [float(true), 0]})
    print('V(pi_b): ',behavior_data.value_of_data(gamma, False), 'V(pi_b) Normalized: ',behavior_data.value_of_data(gamma, True))
    print('V(pi_e): ',eval_data.value_of_data(gamma, False), 'V(pi_e) Normalized: ',eval_data.value_of_data(gamma, True))
    print("KL divergence", divergence)

    get_Qs = getQs(behavior_data, pi_e, processor, env.n_actions)

    for model in models:
        if model == 'MBased_MLE':
            env_model = MaxLikelihoodModel(gamma, max_traj_length=50, action_space_dim=env.n_actions)
            env_model.run(behavior_data)
            Qs_model_based = get_Qs.get(env_model)

            out = estimate(Qs_model_based, behavior_data, gamma, 'Model Based', true)
            dic.update(out)
        elif model == 'MBased_Approx':
            print('*'*20)
            print('Approx estimator not implemented for tabular state space. Please use MBased_MLE instead')
            print('*'*20)
        elif model == 'MFree_Reg':
            DMRegression = DirectMethodRegression(behavior_data, gamma, None, None, None)
            dm_model_ = DMRegression.run(pi_b, pi_e)
            dm_model = QWrapper(dm_model_, {}, is_model=True, modeltype='linear', action_space_dim=env.n_actions)
            Qs_DM_based = get_Qs.get(dm_model)

            out = estimate(Qs_DM_based, behavior_data, gamma,'DM Regression', true)
            dic.update(out)
        elif model == 'MFree_FQE':
            FQE = FittedQEvaluation(behavior_data, gamma)
            out0, Q, mapping = FQE.run(pi_b, pi_e, epsilon=.1, max_epochs=50)
            fqe_model = QWrapper(Q, mapping, is_model=False, action_space_dim=env.n_actions)
            Qs_FQE_based = get_Qs.get(fqe_model)

            out = estimate(Qs_FQE_based, behavior_data, gamma, 'FQE', true)
            dic.update(out)
        elif model == 'MFree_IH':

            ih_max_epochs = None
            matrix_size = None
            inf_horizon = IH(behavior_data, 30, 1e-3, 3e-3, gamma, True, None, env=env)
            inf_hor_output = inf_horizon.evaluate(env, ih_max_epochs, matrix_size)

            # inf_horizon = IH(behavior_data.num_states(), 30, 1e-3, 3e-3, gamma, True, None)
            # inf_hor_output = inf_horizon.evaluate(env, behavior_data)
            inf_hor_output /= 1/np.sum(gamma ** np.arange(max(behavior_data.lengths())))
            dic.update({'IH': [inf_hor_output, (inf_hor_output - true )**2]})
        elif model == 'MFree_MRDR':
            mrdr = MRDR(behavior_data, gamma, modeltype='tabular')
            _ = mrdr.run(pi_e)
            mrdr_model = QWrapper(mrdr, {}, is_model=True, modeltype='linear', action_space_dim=env.n_actions)
            Qs_mrdr_based = get_Qs.get(mrdr_model)

            out = estimate(Qs_mrdr_based, behavior_data, gamma, 'MRDR', true)
            dic.update(out)
        elif model == 'MFree_Retrace_L':
            retrace = Retrace(behavior_data, gamma, lamb=.9, max_iters=50)
            out0, Q, mapping = retrace.run(pi_b, pi_e, 'retrace', epsilon=.002)
            retrace_model = QWrapper(Q, mapping, is_model=False, action_space_dim=env.n_actions)
            Qs_retrace_based = get_Qs.get(retrace_model)

            out = estimate(Qs_retrace_based, behavior_data, gamma, 'Retrace(lambda)', true)
            dic.update(out)

            out0, Q, mapping = retrace.run(pi_b, pi_e, 'tree-backup', epsilon=.002)
            retrace_model = QWrapper(Q, mapping, is_model=False, action_space_dim=env.n_actions)
            Qs_retrace_based = get_Qs.get(retrace_model)

            out = estimate(Qs_retrace_based, behavior_data, gamma, 'Tree-Backup', true)
            dic.update(out)

            out0, Q, mapping = retrace.run(pi_b, pi_e, 'Q^pi(lambda)', epsilon=.002)
            retrace_model = QWrapper(Q, mapping, is_model=False, action_space_dim=env.n_actions)
            Qs_retrace_based = get_Qs.get(retrace_model)

            out = estimate(Qs_retrace_based, behavior_data, gamma, 'Q^pi(lambda)', true)
            dic.update(out)

        elif model == 'IS':
            out = estimate([], behavior_data, gamma, 'IS', true, True)
            dic.update(out)
        else:
            print(model, ' is not a valid method')

        analysis(dic)

    return analysis(dic)

def pixel_gridworld(param, models, debug=False):

    from ope.envs.gridworld import Gridworld
    print(param)
    env = Gridworld(slippage=.2*param['stochastic_env'])

    # policy = {0: 1, 1: 2, 2: 2, 3: 2, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 2, 14: 1, 15: 1, 16: 1, 17: 1, 18: 1, 19: 1, 20: 1, 21: 2, 22: 1, 23: 1, 24: 1, 25: 1, 26: 1, 27: 1, 28: 1, 29: 1, 30: 1, 31: 1, 32: 1, 33: 1, 34: 1, 35: 2, 36: 2, 37: 2, 38: 1, 39: 1, 40: 1, 41: 2, 42: 2, 43: 1, 44: 2, 45: 1, 46: 1, 47: 2, 48: 1, 49: 1, 50: 1, 51: 1, 52: 2, 53: 2, 54: 2, 55: 2, 57: 2, 58: 2, 59: 2, 60: 2, 61: 2, 62: 2, 63: 2}
    # policy = {(key//8, key%8):val for key,val in policy.items()}
    # policy = {(7-key[1], key[0]):val for key,val in policy.items()}
    # policy = {(8*key[0] + key[1]):val for key,val in policy.items()}
    policy = env.best_policy()

    np.random.seed(param['seed'])
    eval_policy = param['eval_policy']/100
    base_policy = param['base_policy']/100
    to_regress_pi_b = param['to_regress_pi_b']
    modeltype = param['modeltype']
    FRAMESKIP = 1
    frameskip = FRAMESKIP
    FRAMEHEIGHT = 1
    gamma = param['gamma']
    assert 0 <= gamma < 1, 'This assumes discounted case. Please make gamma < 1'
    T = param['horizon']

    def to_grid(x):
        x = x.reshape(-1)
        if len(x) > 1: import pdb; pdb.set_trace()
        x = x[0]
        out = np.zeros((8,8))
        if x >= 64:
            return out
        else:
            out[x//8, x%8] = 1.
        return out

    def from_grid(x):
        if len(x.shape) == 3:
            if np.sum(x) == 0:
                x = np.array([64])
            else:
                x = np.array([np.argmax(x.reshape(-1))])
        return x

    processor = lambda x: x
    absorbing_state = processor(np.array([len(policy)]))
    dic = OrderedDict()

    pi_e = EGreedyPolicy(model=TabularPolicy(policy, absorbing=absorbing_state), processor=from_grid, prob_deviation=eval_policy, action_space_dim=env.n_actions)
    pi_b = EGreedyPolicy(model=TabularPolicy(policy, absorbing=absorbing_state), processor=from_grid, prob_deviation=base_policy, action_space_dim=env.n_actions)

    eval_data = rollout(env, pi_e, processor, absorbing_state, N=1024, T=T, frameskip=1, frameheight=1, path=None, filename='tmp',)
    behavior_data = rollout(env, pi_b, processor, absorbing_state, pi_e = pi_e, N=param['num_traj'], T=T, frameskip=1, frameheight=1, path=None, filename='tmp',)



    traj = []
    for trajectory in behavior_data.trajectories:
        frames = []
        for frame in trajectory['frames']:
            frames.append(to_grid(np.array(frame)))
        traj.append(frames)
    for i,frames in enumerate(traj):
        behavior_data.trajectories[i]['frames'] = frames

    if to_regress_pi_b:
        behavior_data.estimate_propensity(True)

    divergence = 0
    dic['KLDivergence'] = divergence

    true = eval_data.value_of_data(gamma, False)
    dic.update({'ON POLICY': [float(true), 0]})
    print('V(pi_b): ',behavior_data.value_of_data(gamma, False), 'V(pi_b) Normalized: ',behavior_data.value_of_data(gamma, True))
    print('V(pi_e): ',eval_data.value_of_data(gamma, False), 'V(pi_e) Normalized: ',eval_data.value_of_data(gamma, True))
    print("KL divergence", divergence)

    get_Qs = getQs(behavior_data, pi_e, processor, env.n_actions)

    for model in models:
        if (model == 'MBased_Approx') or (model == 'MBased_MLE'):
            if model == 'MBased_MLE':
                print('*'*20)
                print('MLE estimator not implemented for continuous state space. Using MBased_Approx instead')
                print('*'*20)
            MBased_max_trajectory_length = 25 if not debug else 1
            batchsize = 32
            mbased_num_epochs = 100 if not debug else 1
            MDPModel = ApproxModel(gamma, None, MBased_max_trajectory_length, FRAMESKIP, FRAMEHEIGHT, processor, action_space_dim=env.n_actions)
            mdpmodel = MDPModel.run(env, behavior_data, mbased_num_epochs, batchsize, 'conv1')

            Qs_model_based = get_Qs.get(mdpmodel)
            out = estimate(Qs_model_based, behavior_data, gamma,'MBased_Approx', true)
            dic.update(out)

        elif model == 'MFree_Reg':
            DMRegression = DirectMethodRegression(behavior_data, gamma, FRAMESKIP, FRAMEHEIGHT, 'conv1', processor)
            dm_max_epochs = 80 if not debug else 1
            _,dm_model_Q = DMRegression.run_NN(env, pi_b, pi_e, dm_max_epochs, epsilon=0.001)

            dm_model = QWrapper(dm_model_Q, None, is_model=True, action_space_dim=env.n_actions, modeltype=modeltype)
            Qs_DM_based = get_Qs.get(dm_model)

            out = estimate(Qs_DM_based, behavior_data, gamma,'DM Regression', true)
            dic.update(out)
        elif model == 'MFree_FQE':
            FQE = FittedQEvaluation(behavior_data, gamma, FRAMESKIP, FRAMEHEIGHT, 'conv1', processor)

            fqe_max_epochs = 80 if not debug else 1
            _,_,fqe_Q = FQE.run_NN(env, pi_b, pi_e, fqe_max_epochs, epsilon=0.0001)

            fqe_model = QWrapper(fqe_Q, None, is_model=True, action_space_dim=env.n_actions, modeltype=modeltype)
            Qs_FQE_based = get_Qs.get(fqe_model)

            out = estimate(Qs_FQE_based, behavior_data, gamma, 'FQE', true)
            dic.update(out)
        elif model == 'MFree_IH':
            ih_max_epochs = 1001 if not debug else 1
            ih_matrix_size = 128
            inf_horizon = IH(behavior_data, 30, 1e-3, 3e-3, gamma, False, 'conv1', processor=processor)
            inf_hor_output = inf_horizon.evaluate(env, ih_max_epochs, ih_matrix_size)
            inf_hor_output /= 1/np.sum(gamma ** np.arange(max(behavior_data.lengths())))
            dic.update({'IH': [inf_hor_output, (inf_hor_output - true )**2]})
        elif model == 'MFree_MRDR':
            mrdr = MRDR(behavior_data, gamma, FRAMESKIP, FRAMEHEIGHT, 'conv1', processor)

            mrdr_max_epochs = 80 if not debug else 1
            mrdr_matrix_size = 1024
            _,_,mrdr_Q = mrdr.run_NN(env, pi_b, pi_e, mrdr_max_epochs, mrdr_matrix_size, epsilon=0.001)
            mrdr_model = QWrapper(mrdr_Q, None, is_model=True, action_space_dim=env.n_actions, modeltype=modeltype)
            Qs_mrdr_based = get_Qs.get(mrdr_model)

            out = estimate(Qs_mrdr_based, behavior_data, gamma, 'MRDR', true)
            dic.update(out)
        elif model == 'MFree_Retrace_L':
            # # print('*'*20)
            # # print('Retrace(lambda) estimator not implemented for continuous state space')
            # # print('*'*20)
            # print('*'*20)
            # print('R(lambda): These methods are incredibly expensive and not as performant. To use, uncomment below.')
            # print('*'*20)
            # pass
            retrace = Retrace(behavior_data, gamma, FRAMESKIP, FRAMEHEIGHT, 'conv1', lamb=.9, processor=processor)

            retrace_max_epochs = 80 if not debug else 1
            _,_,retrace_Q = retrace.run_NN(env, pi_b, pi_e, retrace_max_epochs, 'retrace', epsilon=0.001)
            retrace_model = QWrapper(retrace_Q, None, is_model=True, action_space_dim=env.n_actions, modeltype='conv') # use mlp-based wrapper even for linear
            Qs_retrace_based = get_Qs.get(retrace_model)
            out = estimate(Qs_retrace_based, behavior_data, gamma, 'Retrace(lambda)', true)
            dic.update(out)

            _,_,tree_Q = retrace.run_NN(env, pi_b, pi_e, retrace_max_epochs, 'tree-backup', epsilon=0.001)
            tree_model = QWrapper(tree_Q, None, is_model=True, action_space_dim=env.n_actions, modeltype='conv')
            Qs_tree_based = get_Qs.get(tree_model)
            out = estimate(Qs_tree_based, behavior_data, gamma, 'Tree-Backup', true)
            dic.update(out)

            _,_,q_lambda_Q = retrace.run_NN(env, pi_b, pi_e, retrace_max_epochs, 'Q^pi(lambda)', epsilon=0.001)
            q_lambda_model = QWrapper(q_lambda_Q, None, is_model=True, action_space_dim=env.n_actions, modeltype='conv')
            Qs_q_lambda_based = get_Qs.get(q_lambda_model)
            out = estimate(Qs_q_lambda_based, behavior_data, gamma, 'Q^pi(lambda)', true)
            dic.update(out)

        elif model == 'IS':
            out = estimate([], behavior_data, gamma, 'IS', true, True)
            dic.update(out)
        else:
            print(model, ' is not a valid method')

        analysis(dic)

    return analysis(dic)

def toy_graph(param, models, debug=False):
    from ope.envs.model_fail import ModelFail

    env = ModelFail(make_pomdp=param['is_pomdp'],
                    number_of_pomdp_states = param['pomdp_horizon'],
                    transitions_deterministic=not param['stochastic_env'],
                    max_length = param['horizon'],
                    sparse_rewards = param['sparse_rewards'],
                    stochastic_rewards = param['stochastic_rewards'])

    np.random.seed(param['seed'])
    actions = [0,1]
    eval_policy = param['eval_policy']
    base_policy = param['base_policy']
    to_regress_pi_b = param['to_regress_pi_b']
    gamma = param['gamma']
    assert 0 <= gamma < 1, 'This assumes discounted case. Please make gamma < 1'
    T = param['horizon']
    processor = lambda x: x
    absorbing_state = processor(np.array([env.n_dim-1]))
    dic = OrderedDict()

    assert eval_policy in range(5), 'Eval: Can only choose from 5 policies'
    assert base_policy in range(5), 'Base: Can only choose from 5 policies'
    pi_e = BasicPolicy([0,1], [max(.001, .2*eval_policy), 1-max(.001, .2*eval_policy)])
    pi_b = BasicPolicy([0,1], [max(.001, .2*base_policy), 1-max(.001, .2*base_policy)])

    eval_data = rollout(env, pi_e, processor, absorbing_state, N=max(10000, param['num_traj']), T=T, frameskip=1, frameheight=1, path=None, filename='tmp',)
    behavior_data = rollout(env, pi_b, processor, absorbing_state, pi_e = pi_e, N=param['num_traj'], T=T, frameskip=1, frameheight=1, path=None, filename='tmp',)

    if to_regress_pi_b:
        behavior_data.estimate_propensity()

    true = eval_data.value_of_data(gamma, False)
    dic.update({'ON POLICY': [float(true), 0]})
    print('V(pi_b): ',behavior_data.value_of_data(gamma, False), 'V(pi_b) Normalized: ',behavior_data.value_of_data(gamma, True))
    print('V(pi_e): ',eval_data.value_of_data(gamma, False), 'V(pi_e) Normalized: ',eval_data.value_of_data(gamma, True))

    get_Qs = getQs(behavior_data, pi_e, processor, len(actions))

    for model in models:
        if model == 'MBased_MLE':
            env_model = MaxLikelihoodModel(gamma, max_traj_length=T)
            env_model.run(behavior_data)
            Qs_model_based = get_Qs.get(env_model)

            out = estimate(Qs_model_based, behavior_data, gamma, 'Model Based', true)
            dic.update(out)
        elif model == 'MBased_Approx':
            print('*'*20)
            print('Approx estimator not implemented for tabular state space. Please use MBased_MLE instead')
            print('*'*20)
        elif model == 'MFree_Reg':
            DMRegression = DirectMethodRegression(behavior_data, gamma, None, None, None)
            dm_model_ = DMRegression.run(pi_b, pi_e)
            dm_model = QWrapper(dm_model_, {}, is_model=True, modeltype='linear')
            Qs_DM_based = get_Qs.get(dm_model)

            out = estimate(Qs_DM_based, behavior_data, gamma,'DM Regression', true)
            dic.update(out)
        elif model == 'MFree_FQE':
            FQE = FittedQEvaluation(behavior_data, gamma)
            out0, Q, mapping = FQE.run(pi_b, pi_e)
            fqe_model = QWrapper(Q, mapping, is_model=False)
            Qs_FQE_based = get_Qs.get(fqe_model)

            out = estimate(Qs_FQE_based, behavior_data, gamma, 'FQE', true)
            dic.update(out)
        elif model == 'MFree_IH':
            ih_max_epochs = None
            matrix_size = None
            inf_horizon = IH(behavior_data, 30, 1e-3, 3e-3, gamma, True, None, env=env)
            inf_hor_output = inf_horizon.evaluate(env, ih_max_epochs, matrix_size)
            inf_hor_output /= 1/np.sum(gamma ** np.arange(max(behavior_data.lengths())))
            dic.update({'IH': [inf_hor_output, (inf_hor_output - true )**2]})
        elif model == 'MFree_MRDR':
            mrdr = MRDR(behavior_data, gamma, modeltype = 'tabular')
            _ = mrdr.run(pi_e)
            mrdr_model = QWrapper(mrdr, {}, is_model=True, modeltype='linear') # annoying missname of variable. fix to be modeltype='tabular'
            Qs_mrdr_based = get_Qs.get(mrdr_model)

            out = estimate(Qs_mrdr_based, behavior_data, gamma, 'MRDR', true)
            dic.update(out)
        elif model == 'MFree_Retrace_L':
            retrace = Retrace(behavior_data, gamma, lamb=1.)
            out0, Q, mapping = retrace.run(pi_b, pi_e, 'retrace', epsilon=.001)
            retrace_model = QWrapper(Q, mapping, is_model=False)
            Qs_retrace_based = get_Qs.get(retrace_model)

            out = estimate(Qs_retrace_based, behavior_data, gamma, 'Retrace(lambda)', true)
            dic.update(out)

            out0, Q, mapping = retrace.run(pi_b, pi_e, 'tree-backup', epsilon=.001)
            retrace_model = QWrapper(Q, mapping, is_model=False)
            Qs_retrace_based = get_Qs.get(retrace_model)

            out = estimate(Qs_retrace_based, behavior_data, gamma, 'Tree-Backup', true)
            dic.update(out)

            out0, Q, mapping = retrace.run(pi_b, pi_e, 'Q^pi(lambda)', epsilon=.001)
            retrace_model = QWrapper(Q, mapping, is_model=False)
            Qs_retrace_based = get_Qs.get(retrace_model)

            out = estimate(Qs_retrace_based, behavior_data, gamma, 'Q^pi(lambda)', true)
            dic.update(out)

        elif model == 'IS':
            out = estimate([], behavior_data, gamma, 'IS', true, True)
            dic.update(out)
        else:
            print(model, ' is not a valid method')

        analysis(dic)

    return analysis(dic)

def toy_mc(param, models, debug=False):
    from ope.envs.discrete_toy_mc import DiscreteToyMC
    print(param)
    env = DiscreteToyMC()#n_left = 10, n_right = 10, random_start = False)

    np.random.seed(param['seed'])
    actions = [0,1]
    eval_policy = param['eval_policy']
    base_policy = param['base_policy']
    to_regress_pi_b = param['to_regress_pi_b']
    gamma = param['gamma']
    assert 0 <= gamma < 1, 'This assumes discounted case. Please make gamma < 1'
    T = param['horizon']
    processor = lambda x: x
    absorbing_state = processor(np.array([env.n_dim-1]))
    dic = OrderedDict()

    # assert eval_policy in range(5), 'Eval: Can only choose from 5 policies'
    # assert base_policy in range(5), 'Base: Can only choose from 5 policies'
    pi_e = BasicPolicy([0,1], [1-max(.001, eval_policy/100), max(.001, eval_policy/100)])
    pi_b = BasicPolicy([0,1], [1-max(.001, base_policy/100), max(.001, base_policy/100)])

    eval_data = rollout(env, pi_e, processor, absorbing_state, N=max(1000, param['num_traj']), T=T, frameskip=1, frameheight=1, path=None, filename='tmp',)
    behavior_data = rollout(env, pi_b, processor, absorbing_state, pi_e = pi_e, N=max(0, param['num_traj']), T=T, frameskip=1, frameheight=1, path=None, filename='tmp',)

    if to_regress_pi_b:
        behavior_data.estimate_propensity()

    true = eval_data.value_of_data(gamma, False)
    dic.update({'ON POLICY': [float(true), 0]})
    print('V(pi_b): ',behavior_data.value_of_data(gamma, False), 'V(pi_b) Normalized: ',behavior_data.value_of_data(gamma, True))
    print('V(pi_e): ',eval_data.value_of_data(gamma, False), 'V(pi_e) Normalized: ',eval_data.value_of_data(gamma, True))

    get_Qs = getQs(behavior_data, pi_e, processor, len(actions))

    for model in models:
        if model == 'MBased_MLE':
            env_model = MaxLikelihoodModel(gamma, max_traj_length=50)
            env_model.run(behavior_data)
            Qs_model_based = get_Qs.get(env_model)

            out = estimate(Qs_model_based, behavior_data, gamma, 'Model Based', true)
            dic.update(out)
        elif model == 'MBased_Approx':
            print('*'*20)
            print('Approx estimator not implemented for tabular state space. Please use MBased_MLE instead')
            print('*'*20)
        elif model == 'MFree_Reg':
            DMRegression = DirectMethodRegression(behavior_data, gamma, None, None, None)
            dm_model_ = DMRegression.run(pi_b, pi_e)
            dm_model = QWrapper(dm_model_, {}, is_model=True, modeltype='linear')
            Qs_DM_based = get_Qs.get(dm_model)

            out = estimate(Qs_DM_based, behavior_data, gamma,'DM Regression', true)
            dic.update(out)
        elif model == 'MFree_FQE':
            FQE = FittedQEvaluation(behavior_data, gamma)
            out0, Q, mapping = FQE.run(pi_b, pi_e)
            fqe_model = QWrapper(Q, mapping, is_model=False)
            Qs_FQE_based = get_Qs.get(fqe_model)

            out = estimate(Qs_FQE_based, behavior_data, gamma, 'FQE', true)
            dic.update(out)
        elif model == 'MFree_IH':

            ih_max_epochs = None
            matrix_size = None
            inf_horizon = IH(behavior_data, 30, 1e-3, 3e-3, gamma, True, None, env=env)
            inf_hor_output = inf_horizon.evaluate(env, ih_max_epochs, matrix_size)

            # inf_horizon = IH(behavior_data.num_states(), 30, 1e-3, 3e-3, gamma, True, None)
            # inf_hor_output = inf_horizon.evaluate(env, behavior_data)
            inf_hor_output /= 1/np.sum(gamma ** np.arange(max(behavior_data.lengths())))
            dic.update({'IH': [inf_hor_output, (inf_hor_output - true )**2]})
        elif model == 'MFree_MRDR':
            mrdr = MRDR(behavior_data, gamma, modeltype='tabular')
            _ = mrdr.run(pi_e)
            mrdr_model = QWrapper(mrdr, {}, is_model=True, modeltype='linear')
            Qs_mrdr_based = get_Qs.get(mrdr_model)

            out = estimate(Qs_mrdr_based, behavior_data, gamma, 'MRDR', true)
            dic.update(out)
        elif model == 'MFree_Retrace_L':
            retrace = Retrace(behavior_data, gamma, lamb=.9)
            out0, Q, mapping = retrace.run(pi_b, pi_e, 'retrace', epsilon=.002)
            retrace_model = QWrapper(Q, mapping, is_model=False)
            Qs_retrace_based = get_Qs.get(retrace_model)

            out = estimate(Qs_retrace_based, behavior_data, gamma, 'Retrace(lambda)', true)
            dic.update(out)

            out0, Q, mapping = retrace.run(pi_b, pi_e, 'tree-backup', epsilon=.002)
            retrace_model = QWrapper(Q, mapping, is_model=False)
            Qs_retrace_based = get_Qs.get(retrace_model)

            out = estimate(Qs_retrace_based, behavior_data, gamma, 'Tree-Backup', true)
            dic.update(out)

            out0, Q, mapping = retrace.run(pi_b, pi_e, 'Q^pi(lambda)', epsilon=.002)
            retrace_model = QWrapper(Q, mapping, is_model=False)
            Qs_retrace_based = get_Qs.get(retrace_model)

            out = estimate(Qs_retrace_based, behavior_data, gamma, 'Q^pi(lambda)', true)
            dic.update(out)

        elif model == 'IS':
            out = estimate([], behavior_data, gamma, 'IS', true, True)
            dic.update(out)
        else:
            print(model, ' is not a valid method')

        analysis(dic)

    return analysis(dic)

def mc(param, models, debug=False):
    from ope.envs.modified_mountain_car import ModifiedMountainCarEnv
    FRAMESKIP = 5
    frameskip = FRAMESKIP
    FRAMEHEIGHT = 2
    ABS_STATE = np.array([.5, 0])

    gamma = param['gamma']
    assert 0 <= gamma < 1, 'This assumes discounted case. Please make gamma < 1'

    base_policy = param['base_policy']
    eval_policy = param['eval_policy']
    seed = param['seed']
    N = param['num_traj']
    modeltype = param['modeltype']
    num_trajectories = N
    T = param['horizon']

    np.random.seed(seed)
    env = ModifiedMountainCarEnv(deterministic_start=[-.4, -.5, -.6], seed=seed)

    actions = [0,1,2]
    probs_base = [.01, .1, .25, 1.]
    probs_eval = [0., .1, .25, 1.]
    pi_e = EGreedyPolicy(model=load_model(os.path.join(os.getcwd(),'trained_models','mc_trained_model_Q.h5')), prob_deviation=probs_eval[eval_policy], action_space_dim=len(actions))
    pi_b = EGreedyPolicy(model=load_model(os.path.join(os.getcwd(),'trained_models','mc_trained_model_Q.h5')), prob_deviation=probs_base[base_policy], action_space_dim=len(actions))

    processor = lambda x: x
    absorbing_state = processor(ABS_STATE)
    dic = OrderedDict()

    eval_data = rollout(env, pi_e, processor, absorbing_state, N=param['num_traj'], T=T, frameskip=FRAMESKIP, frameheight=FRAMEHEIGHT, path=None, filename='tmp',)
    behavior_data = rollout(env, pi_b, processor, absorbing_state, pi_e = pi_e, N=param['num_traj'], T=T, frameskip=FRAMESKIP, frameheight=FRAMEHEIGHT, path=None, filename='tmp',)

    true = eval_data.value_of_data(gamma, False)
    dic.update({'ON POLICY': [float(true), 0]})
    print('V(pi_b): ',behavior_data.value_of_data(gamma, False), 'V(pi_b) Normalized: ',behavior_data.value_of_data(gamma, True))
    print('V(pi_e): ',eval_data.value_of_data(gamma, False), 'V(pi_e) Normalized: ',eval_data.value_of_data(gamma, True))

    get_Qs = getQs(behavior_data, pi_e, processor, len(actions))

    ## Test on toy domain
    # def toy_mc(param, models):
    #     from ope.envs.discrete_toy_mc import DiscreteToyMC
    #     print(param)
    #     env = DiscreteToyMC()#n_left = 10, n_right = 10, random_start = False)

    #     FRAMESKIP = 1
    #     frameskip = 1
    #     FRAMEHEIGHT = 1

    #     np.random.seed(param['seed'])
    #     actions = [0,1]
    #     eval_policy = param['eval_policy']
    #     base_policy = param['base_policy']
    #     modeltype = param['modeltype']
    #     gamma = param['gamma']
    #     assert 0 <= gamma < 1, 'This assumes discounted case. Please make gamma < 1'
    #     T = param['horizon']
    #     processor = lambda x: x
    #     absorbing_state = processor(np.array([0]))
    #     dic = OrderedDict()

    #     # assert eval_policy in range(5), 'Eval: Can only choose from 5 policies'
    #     # assert base_policy in range(5), 'Base: Can only choose from 5 policies'
    #     pi_e = BasicPolicy([0,1], [1-max(.001, eval_policy/100), max(.001, eval_policy/100)])
    #     pi_b = BasicPolicy([0,1], [1-max(.001, base_policy/100), max(.001, base_policy/100)])

    #     eval_data = rollout(env, pi_e, processor, absorbing_state, N=max(128, param['num_traj']), T=T, frameskip=1, frameheight=1, path=None, filename='tmp',)
    #     behavior_data = rollout(env, pi_b, processor, absorbing_state, pi_e = pi_e, N=param['num_traj'], T=T, frameskip=1, frameheight=1, path=None, filename='tmp',)

    #     true = eval_data.value_of_data(gamma, False)
    #     dic.update({'ON POLICY': [float(true), 0]})
    #     print('V(pi_b): ',behavior_data.value_of_data(gamma, False), 'V(pi_b) Normalized: ',behavior_data.value_of_data(gamma, True))
    #     print('V(pi_e): ',eval_data.value_of_data(gamma, False), 'V(pi_e) Normalized: ',eval_data.value_of_data(gamma, True))

    #     get_Qs = getQs(behavior_data, pi_e, processor, env.n_actions)

    for model in models:
        if (model == 'MBased_Approx') or (model == 'MBased_MLE'):
            if model == 'MBased_MLE':
                print('*'*20)
                print('MLE estimator not implemented for continuous state space. Using MBased_Approx instead')
                print('*'*20)
            MBased_max_trajectory_length = 50 if not debug else 1
            batchsize = 32
            mbased_num_epochs = 100
            MDPModel = ApproxModel(gamma, None, MBased_max_trajectory_length, FRAMESKIP, FRAMEHEIGHT)
            mdpmodel = MDPModel.run(env, behavior_data, mbased_num_epochs, batchsize, modeltype)

            Qs_model_based = get_Qs.get(mdpmodel)
            out = estimate(Qs_model_based, behavior_data, gamma,'MBased_Approx', true)
            dic.update(out)

        elif model == 'MFree_Reg':
            DMRegression = DirectMethodRegression(behavior_data, gamma, FRAMESKIP, FRAMEHEIGHT, modeltype)
            dm_max_epochs = 80 if not debug else 1
            if modeltype == 'linear':
                dm_model_Q = DMRegression.run_linear(env, pi_b, pi_e, dm_max_epochs, epsilon=0.001)
            else:
                _,dm_model_Q = DMRegression.run_NN(env, pi_b, pi_e, dm_max_epochs, epsilon=0.001)

            dm_model = QWrapper(dm_model_Q, None, is_model=True, action_space_dim=env.n_actions, modeltype=modeltype)
            Qs_DM_based = get_Qs.get(dm_model)

            out = estimate(Qs_DM_based, behavior_data, gamma,'DM Regression', true)
            dic.update(out)
        elif model == 'MFree_FQE':
            FQE = FittedQEvaluation(behavior_data, gamma, FRAMESKIP, FRAMEHEIGHT, modeltype)

            fqe_max_epochs = 160 if not debug else 1
            if modeltype == 'linear':
                fqe_Q = FQE.run_linear(env, pi_b, pi_e, fqe_max_epochs, epsilon=0.001)
            else:
                _,_,fqe_Q = FQE.run_NN(env, pi_b, pi_e, fqe_max_epochs, epsilon=0.001)

            fqe_model = QWrapper(fqe_Q, None, is_model=True, action_space_dim=env.n_actions, modeltype=modeltype)
            Qs_FQE_based = get_Qs.get(fqe_model)

            out = estimate(Qs_FQE_based, behavior_data, gamma, 'FQE', true)
            dic.update(out)
        elif model == 'MFree_IH':
            ih_max_epochs = 10001 if not debug else 1
            ih_matrix_size = 1024
            inf_horizon = IH(behavior_data, 30, 1e-3, 3e-3, gamma, False, modeltype)
            inf_hor_output = inf_horizon.evaluate(env, ih_max_epochs, ih_matrix_size)
            inf_hor_output /= 1/np.sum(gamma ** np.arange(max(behavior_data.lengths())))
            dic.update({'IH': [inf_hor_output, (inf_hor_output - true )**2]})
        elif model == 'MFree_MRDR':
            mrdr = MRDR(behavior_data, gamma, FRAMESKIP, FRAMEHEIGHT, modeltype)

            mrdr_max_epochs = 80 if not debug else 1
            mrdr_matrix_size = 1024

            if modeltype == 'linear':
                mrdr_Q = mrdr.run(pi_e)
            else:
                _,_,mrdr_Q = mrdr.run_NN(env, pi_b, pi_e, mrdr_max_epochs, mrdr_matrix_size, epsilon=0.001)

            mrdr_model = QWrapper(mrdr_Q, None, is_model=True, action_space_dim=env.n_actions, modeltype=modeltype)
            Qs_mrdr_based = get_Qs.get(mrdr_model)

            out = estimate(Qs_mrdr_based, behavior_data, gamma, 'MRDR', true)
            dic.update(out)
        elif model == 'MFree_Retrace_L':
            # print('*'*20)
            # print('Retrace(lambda) estimator not implemented for continuous state space')
            # print('*'*20)
            retrace = Retrace(behavior_data, gamma, FRAMESKIP, FRAMEHEIGHT, modeltype, lamb=.9)

            retrace_max_epochs = 80 if not debug else 1
            _,_,retrace_Q = retrace.run_NN(env, pi_b, pi_e, retrace_max_epochs, 'retrace', epsilon=0.001)
            retrace_model = QWrapper(retrace_Q, None, is_model=True, action_space_dim=env.n_actions, modeltype='mlp') # use mlp-based wrapper even for linear
            Qs_retrace_based = get_Qs.get(retrace_model)
            out = estimate(Qs_retrace_based, behavior_data, gamma, 'Retrace(lambda)', true)
            dic.update(out)

            _,_,tree_Q = retrace.run_NN(env, pi_b, pi_e, retrace_max_epochs, 'tree-backup', epsilon=0.001)
            tree_model = QWrapper(tree_Q, None, is_model=True, action_space_dim=env.n_actions, modeltype='mlp')
            Qs_tree_based = get_Qs.get(tree_model)
            out = estimate(Qs_tree_based, behavior_data, gamma, 'Tree-Backup', true)
            dic.update(out)

            _,_,q_lambda_Q = retrace.run_NN(env, pi_b, pi_e, retrace_max_epochs, 'Q^pi(lambda)', epsilon=0.001)
            q_lambda_model = QWrapper(q_lambda_Q, None, is_model=True, action_space_dim=env.n_actions, modeltype='mlp')
            Qs_q_lambda_based = get_Qs.get(q_lambda_model)
            out = estimate(Qs_q_lambda_based, behavior_data, gamma, 'Q^pi(lambda)', true)
            dic.update(out)


        elif model == 'IS':
            out = estimate([], behavior_data, gamma, 'IS', true, True)
            dic.update(out)
        else:
            print(model, ' is not a valid method')

        analysis(dic)

    return analysis(dic)

def pixel_mc(param, models, debug=False):
    from ope.envs.modified_mountain_car import ModifiedMountainCarEnv
    FRAMESKIP = 5
    frameskip = FRAMESKIP
    FRAMEHEIGHT = 2
    ABS_STATE = np.array([.5, 0])

    gamma = param['gamma']
    assert 0 <= gamma < 1, 'This assumes discounted case. Please make gamma < 1'

    base_policy = param['base_policy']
    eval_policy = param['eval_policy']
    seed = param['seed']
    N = param['num_traj']
    modeltype = param['modeltype']
    num_trajectories = N
    T = param['horizon']

    np.random.seed(seed)
    env = ModifiedMountainCarEnv(deterministic_start=[-.4, -.5, -.6], seed=seed)

    actions = [0,1,2]
    probs_base = [.01, .1, .25, 1.]
    probs_eval = [0., .1, .25, 1.]
    pi_e = EGreedyPolicy(model=load_model(os.path.join(os.getcwd(),'ope','trained_models','mc_pixel_trained_model_Q.h5')), prob_deviation=probs_eval[eval_policy], action_space_dim=len(actions))
    pi_b = EGreedyPolicy(model=load_model(os.path.join(os.getcwd(),'ope','trained_models','mc_pixel_trained_model_Q.h5')), prob_deviation=probs_base[base_policy], action_space_dim=len(actions))

    processor = lambda x: env.pos_to_image(x, True)
    absorbing_state = ABS_STATE
    dic = OrderedDict()

    eval_data = rollout(env, pi_e, processor, absorbing_state, N=param['num_traj'], T=T, frameskip=FRAMESKIP, frameheight=FRAMEHEIGHT, path=None, filename='tmp',)
    behavior_data = rollout(env, pi_b, processor, absorbing_state, pi_e = pi_e, N=param['num_traj'], T=T, frameskip=FRAMESKIP, frameheight=FRAMEHEIGHT, path=None, filename='tmp',)

    true = eval_data.value_of_data(gamma, False)
    dic.update({'ON POLICY': [float(true), 0]})
    print('V(pi_b): ',behavior_data.value_of_data(gamma, False), 'V(pi_b) Normalized: ',behavior_data.value_of_data(gamma, True))
    print('V(pi_e): ',eval_data.value_of_data(gamma, False), 'V(pi_e) Normalized: ',eval_data.value_of_data(gamma, True))

    get_Qs = getQs(behavior_data, pi_e, processor, len(actions))

    for model in models:
        if (model == 'MBased_Approx') or (model == 'MBased_MLE'):
            if model == 'MBased_MLE':
                print('*'*20)
                print('MLE estimator not implemented for continuous state space. Using MBased_Approx instead')
                print('*'*20)
            MBased_max_trajectory_length = 50 if not debug else 1
            batchsize = 32
            mbased_num_epochs = 100 if not debug else 1
            MDPModel = ApproxModel(gamma, None, MBased_max_trajectory_length, FRAMESKIP, FRAMEHEIGHT, processor)
            mdpmodel = MDPModel.run(env, behavior_data, mbased_num_epochs, batchsize, modeltype)

            Qs_model_based = get_Qs.get(mdpmodel)
            out = estimate(Qs_model_based, behavior_data, gamma,'MBased_Approx', true)
            dic.update(out)

        elif model == 'MFree_Reg':
            DMRegression = DirectMethodRegression(behavior_data, gamma, FRAMESKIP, FRAMEHEIGHT, modeltype, processor)
            dm_max_epochs = 80 if not debug else 1
            _,dm_model_Q = DMRegression.run_NN(env, pi_b, pi_e, dm_max_epochs, epsilon=0.001)

            dm_model = QWrapper(dm_model_Q, None, is_model=True, action_space_dim=env.n_actions, modeltype=modeltype)
            Qs_DM_based = get_Qs.get(dm_model)

            out = estimate(Qs_DM_based, behavior_data, gamma,'DM Regression', true)
            dic.update(out)
        elif model == 'MFree_FQE':
            FQE = FittedQEvaluation(behavior_data, gamma, FRAMESKIP, FRAMEHEIGHT, modeltype, processor)

            fqe_max_epochs = 160 if not debug else 1
            _,_,fqe_Q = FQE.run_NN(env, pi_b, pi_e, fqe_max_epochs, epsilon=0.001)

            fqe_model = QWrapper(fqe_Q, None, is_model=True, action_space_dim=env.n_actions, modeltype=modeltype)
            Qs_FQE_based = get_Qs.get(fqe_model)

            out = estimate(Qs_FQE_based, behavior_data, gamma, 'FQE', true)
            dic.update(out)
        elif model == 'MFree_IH':
            ih_max_epochs = 10001 if not debug else 1
            ih_matrix_size = 128
            inf_horizon = IH(behavior_data, 30, 1e-3, 3e-3, gamma, False, modeltype, processor=processor)
            inf_hor_output = inf_horizon.evaluate(env, ih_max_epochs, ih_matrix_size)
            inf_hor_output /= 1/np.sum(gamma ** np.arange(max(behavior_data.lengths())))
            dic.update({'IH': [inf_hor_output, (inf_hor_output - true )**2]})
        elif model == 'MFree_MRDR':
            mrdr = MRDR(behavior_data, gamma, FRAMESKIP, FRAMEHEIGHT, modeltype, processor)

            mrdr_max_epochs = 80 if not debug else 1
            mrdr_matrix_size = 1024
            _,_,mrdr_Q = mrdr.run_NN(env, pi_b, pi_e, mrdr_max_epochs, mrdr_matrix_size, epsilon=0.001)
            mrdr_model = QWrapper(mrdr_Q, None, is_model=True, action_space_dim=env.n_actions, modeltype=modeltype)
            Qs_mrdr_based = get_Qs.get(mrdr_model)

            out = estimate(Qs_mrdr_based, behavior_data, gamma, 'MRDR', true)
            dic.update(out)
        elif model == 'MFree_Retrace_L':
            # print('*'*20)
            # print('Retrace(lambda) estimator not implemented for continuous state space')
            # print('*'*20)
            print('*'*20)
            print('R(lambda): These methods are incredibly expensive and not as performant. To use, uncomment below.')
            print('*'*20)
            pass
            # retrace = Retrace(behavior_data, gamma, FRAMESKIP, FRAMEHEIGHT, modeltype, lamb=.9, processor=processor)

            # retrace_max_epochs = 80 if not debug else 1
            # _,_,retrace_Q = retrace.run_NN(env, pi_b, pi_e, retrace_max_epochs, 'retrace', epsilon=0.001)
            # retrace_model = QWrapper(retrace_Q, None, is_model=True, action_space_dim=env.n_actions, modeltype='mlp') # use mlp-based wrapper even for linear
            # Qs_retrace_based = get_Qs.get(retrace_model)
            # out = estimate(Qs_retrace_based, behavior_data, gamma, 'Retrace(lambda)', true)
            # dic.update(out)

            # _,_,tree_Q = retrace.run_NN(env, pi_b, pi_e, retrace_max_epochs, 'tree-backup', epsilon=0.001)
            # tree_model = QWrapper(tree_Q, None, is_model=True, action_space_dim=env.n_actions, modeltype='mlp')
            # Qs_tree_based = get_Qs.get(tree_model)
            # out = estimate(Qs_tree_based, behavior_data, gamma, 'Tree-Backup', true)
            # dic.update(out)

            # _,_,q_lambda_Q = retrace.run_NN(env, pi_b, pi_e, retrace_max_epochs, 'Q^pi(lambda)', epsilon=0.001)
            # q_lambda_model = QWrapper(q_lambda_Q, None, is_model=True, action_space_dim=env.n_actions, modeltype='mlp')
            # Qs_q_lambda_based = get_Qs.get(q_lambda_model)
            # out = estimate(Qs_q_lambda_based, behavior_data, gamma, 'Q^pi(lambda)', true)
            # dic.update(out)

        elif model == 'IS':
            out = estimate([], behavior_data, gamma, 'IS', true, True)
            dic.update(out)
        else:
            print(model, ' is not a valid method')

        analysis(dic)

    return analysis(dic)

def breakout(param, models, debug=False):
    import gym
    from PIL import Image
    FRAMESKIP = 1
    frameskip = FRAMESKIP
    FRAMEHEIGHT = 4
    NO_OP_STEPS = 3

    frame_shape = (84, 84)
    window_length = FRAMEHEIGHT
    input_shape = (window_length,) + frame_shape
    ABS_STATE = np.zeros(frame_shape).astype('uint8')

    gamma = param['gamma']
    assert 0 <= gamma < 1, 'This assumes discounted case. Please make gamma < 1'

    base_policy = param['base_policy']
    eval_policy = param['eval_policy']
    seed = param['seed']
    N = param['num_traj']
    modeltype = param['modeltype']
    num_trajectories = N
    T = param['horizon']

    env = gym.make('BreakoutDeterministic-v4')
    np.random.seed(seed)
    env.seed(seed)
    env.action_space.n = 3
    nb_actions = 3
    env.n_actions = env.action_space.n
    env.n_dim = 84
    env.save_as_int = True

    from keras.models import Sequential, Model
    from keras.layers import Dense, Activation, Flatten, Dropout, concatenate, Permute
    from keras.layers import Input, Conv2D
    from keras.optimizers import Adam
    from keras.activations import relu, linear
    from keras.layers.advanced_activations import LeakyReLU

    action_map = {0:1, 1:2, 'default': 3}
    model = Sequential()
    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=(4, 84, 84)))
    model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(3))
    model.summary()
    model.load_weights(os.path.join(os.getcwd(),'trained_models','breakout.h5'))

    actions = range(nb_actions)
    probs_base = [.01, .1, .5, 1.]
    probs_eval = [.01, .1, .5, 1.]
    pi_e = EGreedyPolicy(model=model, prob_deviation=probs_eval[eval_policy], action_space_dim=len(actions), action_map=action_map)
    pi_b = EGreedyPolicy(model=model, prob_deviation=probs_base[base_policy], action_space_dim=len(actions), action_map=action_map)

    # def preprocessor(img):
    #     img = Image.fromarray(img[0])
    #     img = np.array(img.resize(frame_shape).convert('L'))
    #     img = img.astype('uint8')

    #     return img

    def preprocessor(img):
        processed_observe = np.uint8(resize(rgb2gray(img)[0], (84, 84), mode='constant') * 255)
        return processed_observe

    def processor(img):
        return img.astype('float32') / 255.

    absorbing_state = ABS_STATE
    dic = OrderedDict()

    eval_data = rollout(env, pi_e, processor, absorbing_state, N=param['num_traj'], T=T, frameskip=FRAMESKIP, frameheight=FRAMEHEIGHT, no_op_steps = NO_OP_STEPS, path=None, filename='tmp',preprocessor=preprocessor, visualize='pi_e')
    behavior_data = rollout(env, pi_b, processor, absorbing_state, pi_e = pi_e, N=param['num_traj'], T=T, frameskip=FRAMESKIP, frameheight=FRAMEHEIGHT, no_op_steps = NO_OP_STEPS, path=None, filename='tmp',preprocessor=preprocessor, visualize='pi_b')

    true = eval_data.value_of_data(gamma, False)
    dic.update({'ON POLICY': [float(true), 0]})
    print('V(pi_b): ',behavior_data.value_of_data(gamma, False), 'V(pi_b) Normalized: ',behavior_data.value_of_data(gamma, True))
    print('V(pi_e): ',eval_data.value_of_data(gamma, False), 'V(pi_e) Normalized: ',eval_data.value_of_data(gamma, True))

    get_Qs = getQs(behavior_data, pi_e, processor, len(actions))

    for model in models:
        if (model == 'MBased_Approx') or (model == 'MBased_MLE'):
            # if model == 'MBased_MLE':
            #     print('*'*20)
            #     print('MLE estimator not implemented for continuous state space. Using MBased_Approx instead')
            #     print('*'*20)
            # print('*'*20)
            # print('R(lambda): These methods are incredibly expensive and not as performant. To use, uncomment below.')
            # print('*'*20)
            # pass
            MBased_max_trajectory_length = 50 if not debug else 1
            batchsize = 32
            mbased_num_epochs = 100 if not debug else 1
            MDPModel = ApproxModel(gamma, None, MBased_max_trajectory_length, FRAMESKIP, FRAMEHEIGHT, processor, action_space_dim=env.n_actions)
            mdpmodel = MDPModel.run(env, behavior_data, mbased_num_epochs, batchsize, modeltype)

            Qs_model_based = get_Qs.get(mdpmodel)
            out = estimate(Qs_model_based, behavior_data, gamma,'MBased_Approx', true)
            dic.update(out)

        elif model == 'MFree_Reg':
            DMRegression = DirectMethodRegression(behavior_data, gamma, FRAMESKIP, FRAMEHEIGHT, modeltype, processor)
            dm_max_epochs = 80 if not debug else 2
            _,dm_model_Q = DMRegression.run_NN(env, pi_b, pi_e, dm_max_epochs, epsilon=0.001)

            dm_model = QWrapper(dm_model_Q, None, is_model=True, action_space_dim=env.n_actions, modeltype=modeltype)
            Qs_DM_based = get_Qs.get(dm_model)

            out = estimate(Qs_DM_based, behavior_data, gamma,'DM Regression', true)
            dic.update(out)
        elif model == 'MFree_FQE':
            FQE = FittedQEvaluation(behavior_data, gamma, FRAMESKIP, FRAMEHEIGHT, modeltype, processor)

            fqe_max_epochs = 600 if not debug else 1
            _,_,fqe_Q = FQE.run_NN(env, pi_b, pi_e, fqe_max_epochs, epsilon=0.001, perc_of_dataset=.3)

            fqe_model = QWrapper(fqe_Q, None, is_model=True, action_space_dim=env.n_actions, modeltype=modeltype)
            Qs_FQE_based = get_Qs.get(fqe_model)

            out = estimate(Qs_FQE_based, behavior_data, gamma, 'FQE', true)
            dic.update(out)
        elif model == 'MFree_IH':
            ih_max_epochs = 10001 if not debug else 1
            ih_matrix_size = 128
            inf_horizon = IH(behavior_data, 30, 1e-3, 3e-3, gamma, False, modeltype, processor=processor)
            inf_hor_output = inf_horizon.evaluate(env, ih_max_epochs, ih_matrix_size)
            inf_hor_output /= 1/np.sum(gamma ** np.arange(max(behavior_data.lengths())))
            dic.update({'IH': [inf_hor_output, (inf_hor_output - true )**2]})
        elif model == 'MFree_MRDR':
            mrdr = MRDR(behavior_data, gamma, FRAMESKIP, FRAMEHEIGHT, modeltype, processor)

            mrdr_max_epochs = 80 if not debug else 1
            mrdr_matrix_size = 1024
            _,_,mrdr_Q = mrdr.run_NN(env, pi_b, pi_e, mrdr_max_epochs, mrdr_matrix_size, epsilon=0.001)
            mrdr_model = QWrapper(mrdr_Q, None, is_model=True, action_space_dim=env.n_actions, modeltype=modeltype)
            Qs_mrdr_based = get_Qs.get(mrdr_model)

            out = estimate(Qs_mrdr_based, behavior_data, gamma, 'MRDR', true)
            dic.update(out)
        elif model == 'MFree_Retrace_L':
            print('*'*20)
            print('R(lambda): These methods are incredibly expensive and not as performant. To use, uncomment below.')
            print('*'*20)
            pass
            # # print('*'*20)
            # # print('Retrace(lambda) estimator not implemented for continuous state space')
            # # print('*'*20)
            # retrace = Retrace(behavior_data, gamma, FRAMESKIP, FRAMEHEIGHT, modeltype, lamb=.9, processor=processor)

            # retrace_max_epochs = 80 if not debug else 1
            # _,_,retrace_Q = retrace.run_NN(env, pi_b, pi_e, retrace_max_epochs, 'retrace', epsilon=0.001)
            # retrace_model = QWrapper(retrace_Q, None, is_model=True, action_space_dim=env.n_actions, modeltype='mlp') # use mlp-based wrapper even for linear
            # Qs_retrace_based = get_Qs.get(retrace_model)
            # out = estimate(Qs_retrace_based, behavior_data, gamma, 'Retrace(lambda)', true)
            # dic.update(out)

            # _,_,tree_Q = retrace.run_NN(env, pi_b, pi_e, retrace_max_epochs, 'tree-backup', epsilon=0.001)
            # tree_model = QWrapper(tree_Q, None, is_model=True, action_space_dim=env.n_actions, modeltype='mlp')
            # Qs_tree_based = get_Qs.get(tree_model)
            # out = estimate(Qs_tree_based, behavior_data, gamma, 'Tree-Backup', true)
            # dic.update(out)

            # _,_,q_lambda_Q = retrace.run_NN(env, pi_b, pi_e, retrace_max_epochs, 'Q^pi(lambda)', epsilon=0.001)
            # q_lambda_model = QWrapper(q_lambda_Q, None, is_model=True, action_space_dim=env.n_actions, modeltype='mlp')
            # Qs_q_lambda_based = get_Qs.get(q_lambda_model)
            # out = estimate(Qs_q_lambda_based, behavior_data, gamma, 'Q^pi(lambda)', true)
            # dic.update(out)

        elif model == 'IS':
            out = estimate([], behavior_data, gamma, 'IS', true, True)
            dic.update(out)
        else:
            print(model, ' is not a valid method')

        analysis(dic)

    return analysis(dic)

def enduro(param, models, debug=False):
    import gym
    from PIL import Image
    FRAMESKIP = 1
    frameskip = FRAMESKIP
    FRAMEHEIGHT = 4

    frame_shape = (84, 84)
    window_length = FRAMEHEIGHT
    input_shape = (window_length,) + frame_shape
    ABS_STATE = np.zeros(frame_shape).astype('uint8')

    gamma = param['gamma']
    assert 0 <= gamma < 1, 'This assumes discounted case. Please make gamma < 1'

    base_policy = param['base_policy']
    eval_policy = param['eval_policy']
    seed = param['seed']
    N = param['num_traj']
    modeltype = param['modeltype']
    num_trajectories = N
    T = param['horizon']

    env = gym.make('Enduro-v0')
    np.random.seed(seed)
    env.seed(seed)
    env.n_actions = env.action_space.n
    env.n_dim = 128
    env.save_as_int = True

    from keras.models import Sequential, Model
    from keras.layers import Dense, Activation, Flatten, Dropout, concatenate, Permute
    from keras.layers import Input, Conv2D
    from keras.optimizers import Adam
    from keras.activations import relu, linear
    from keras.layers.advanced_activations import LeakyReLU

    nb_actions = env.action_space.n
    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=input_shape))
    model.add(Conv2D(32, (8, 8), strides=(4, 4)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    model.load_weights(os.path.join(os.getcwd(),'trained_models','enduro.h5f'))


    actions = range(nb_actions)
    probs_base = [.01, .1, .25, 1.]
    probs_eval = [.01, .1, .25, 1.]
    pi_e = EGreedyPolicy(model=model, prob_deviation=probs_eval[eval_policy], action_space_dim=len(actions))
    pi_b = EGreedyPolicy(model=model, prob_deviation=probs_base[base_policy], action_space_dim=len(actions))

    def preprocessor(img):
        img = Image.fromarray(img[0])
        img = np.array(img.resize(frame_shape).convert('L'))
        img = img.astype('uint8')

        return img

    def processor(img):
        return img.astype('float32') / 255.

    absorbing_state = ABS_STATE
    dic = OrderedDict()

    eval_data = rollout(env, pi_e, processor, absorbing_state, N=param['num_traj'], T=T, frameskip=FRAMESKIP, frameheight=FRAMEHEIGHT, path=None, filename='tmp',preprocessor=preprocessor, visualize=None)
    behavior_data = rollout(env, pi_b, processor, absorbing_state, pi_e = pi_e, N=param['num_traj'], T=T, frameskip=FRAMESKIP, frameheight=FRAMEHEIGHT, path=None, filename='tmp',preprocessor=preprocessor, visualize=None)

    true = eval_data.value_of_data(gamma, False)
    dic.update({'ON POLICY': [float(true), 0]})
    print('V(pi_b): ',behavior_data.value_of_data(gamma, False), 'V(pi_b) Normalized: ',behavior_data.value_of_data(gamma, True))
    print('V(pi_e): ',eval_data.value_of_data(gamma, False), 'V(pi_e) Normalized: ',eval_data.value_of_data(gamma, True))

    get_Qs = getQs(behavior_data, pi_e, processor, len(actions))

    for model in models:
        if (model == 'MBased_Approx') or (model == 'MBased_MLE'):
            if model == 'MBased_MLE':
                print('*'*20)
                print('MLE estimator not implemented for continuous state space. Using MBased_Approx instead')
                print('*'*20)
            print('*'*20)
            print('R(lambda): These methods are incredibly expensive and not as performant. To use, uncomment below.')
            print('*'*20)
            pass
            # MBased_max_trajectory_length = 50 if not debug else 1
            # batchsize = 32
            # mbased_num_epochs = 100 if not debug else 1
            # MDPModel = ApproxModel(gamma, None, MBased_max_trajectory_length, FRAMESKIP, FRAMEHEIGHT, processor, action_space_dim=env.n_actions)
            # mdpmodel = MDPModel.run(env, behavior_data, mbased_num_epochs, batchsize, modeltype)

            # Qs_model_based = get_Qs.get(mdpmodel)
            # out = estimate(Qs_model_based, behavior_data, gamma,'MBased_Approx', true)
            # dic.update(out)

        elif model == 'MFree_Reg':
            DMRegression = DirectMethodRegression(behavior_data, gamma, FRAMESKIP, FRAMEHEIGHT, modeltype, processor)
            dm_max_epochs = 80 if not debug else 2
            _,dm_model_Q = DMRegression.run_NN(env, pi_b, pi_e, dm_max_epochs, epsilon=0.001)

            dm_model = QWrapper(dm_model_Q, None, is_model=True, action_space_dim=env.n_actions, modeltype=modeltype)
            Qs_DM_based = get_Qs.get(dm_model)

            out = estimate(Qs_DM_based, behavior_data, gamma,'DM Regression', true)
            dic.update(out)
        elif model == 'MFree_FQE':
            FQE = FittedQEvaluation(behavior_data, gamma, FRAMESKIP, FRAMEHEIGHT, modeltype, processor)

            fqe_max_epochs = 600 if not debug else 1
            _,_,fqe_Q = FQE.run_NN(env, pi_b, pi_e, fqe_max_epochs, epsilon=0.001, perc_of_dataset=.03)

            fqe_model = QWrapper(fqe_Q, None, is_model=True, action_space_dim=env.n_actions, modeltype=modeltype)
            Qs_FQE_based = get_Qs.get(fqe_model)

            out = estimate(Qs_FQE_based, behavior_data, gamma, 'FQE', true)
            dic.update(out)
        elif model == 'MFree_IH':
            ih_max_epochs = 10001 if not debug else 1
            ih_matrix_size = 128
            inf_horizon = IH(behavior_data, 30, 1e-3, 3e-3, gamma, False, modeltype, processor=processor)
            inf_hor_output = inf_horizon.evaluate(env, ih_max_epochs, ih_matrix_size)
            inf_hor_output /= 1/np.sum(gamma ** np.arange(max(behavior_data.lengths())))
            dic.update({'IH': [inf_hor_output, (inf_hor_output - true )**2]})
        elif model == 'MFree_MRDR':
            mrdr = MRDR(behavior_data, gamma, FRAMESKIP, FRAMEHEIGHT, modeltype, processor)

            mrdr_max_epochs = 80 if not debug else 1
            mrdr_matrix_size = 1024
            _,_,mrdr_Q = mrdr.run_NN(env, pi_b, pi_e, mrdr_max_epochs, mrdr_matrix_size, epsilon=0.001)
            mrdr_model = QWrapper(mrdr_Q, None, is_model=True, action_space_dim=env.n_actions, modeltype=modeltype)
            Qs_mrdr_based = get_Qs.get(mrdr_model)

            out = estimate(Qs_mrdr_based, behavior_data, gamma, 'MRDR', true)
            dic.update(out)
        elif model == 'MFree_Retrace_L':
            print('*'*20)
            print('R(lambda): These methods are incredibly expensive and not as performant. To use, uncomment below.')
            print('*'*20)
            pass
            # # print('*'*20)
            # # print('Retrace(lambda) estimator not implemented for continuous state space')
            # # print('*'*20)
            # retrace = Retrace(behavior_data, gamma, FRAMESKIP, FRAMEHEIGHT, modeltype, lamb=.9, processor=processor)

            # retrace_max_epochs = 80 if not debug else 1
            # _,_,retrace_Q = retrace.run_NN(env, pi_b, pi_e, retrace_max_epochs, 'retrace', epsilon=0.001)
            # retrace_model = QWrapper(retrace_Q, None, is_model=True, action_space_dim=env.n_actions, modeltype='mlp') # use mlp-based wrapper even for linear
            # Qs_retrace_based = get_Qs.get(retrace_model)
            # out = estimate(Qs_retrace_based, behavior_data, gamma, 'Retrace(lambda)', true)
            # dic.update(out)

            # _,_,tree_Q = retrace.run_NN(env, pi_b, pi_e, retrace_max_epochs, 'tree-backup', epsilon=0.001)
            # tree_model = QWrapper(tree_Q, None, is_model=True, action_space_dim=env.n_actions, modeltype='mlp')
            # Qs_tree_based = get_Qs.get(tree_model)
            # out = estimate(Qs_tree_based, behavior_data, gamma, 'Tree-Backup', true)
            # dic.update(out)

            # _,_,q_lambda_Q = retrace.run_NN(env, pi_b, pi_e, retrace_max_epochs, 'Q^pi(lambda)', epsilon=0.001)
            # q_lambda_model = QWrapper(q_lambda_Q, None, is_model=True, action_space_dim=env.n_actions, modeltype='mlp')
            # Qs_q_lambda_based = get_Qs.get(q_lambda_model)
            # out = estimate(Qs_q_lambda_based, behavior_data, gamma, 'Q^pi(lambda)', true)
            # dic.update(out)

        elif model == 'IS':
            out = estimate([], behavior_data, gamma, 'IS', true, True)
            dic.update(out)
        else:
            print(model, ' is not a valid method')

        analysis(dic)

    return analysis(dic)

def baird(param, models, debug=False):
    from ope.envs.baird import Baird
    print(param)
    env = Baird()

    FRAMESKIP = 1
    frameskip = FRAMESKIP
    FRAMEHEIGHT = 1
    ABS_STATE = np.array([env.terminal])


    modeltype = param['modeltype']
    np.random.seed(param['seed'])
    actions = [0,1]
    eval_policy = param['eval_policy']
    base_policy = param['base_policy']
    gamma = param['gamma']
    assert 0 <= gamma < 1, 'This assumes discounted case. Please make gamma < 1'
    T = param['horizon']

    def preprocessor(x): return env.processor(x[0])
    processor = lambda x: x
    absorbing_state = ABS_STATE

    dic = OrderedDict()

    pi_e = BasicPolicy([0,1], [0, 1])
    pi_b = BasicPolicy([0,1], [6/7, 1/7])

    eval_data = rollout(env, pi_e, processor, absorbing_state, N=max(128, param['num_traj']), T=T, frameskip=1, frameheight=1, path=None, filename='tmp',preprocessor=preprocessor)
    behavior_data = rollout(env, pi_b, processor, absorbing_state, pi_e = pi_e, N=param['num_traj'], T=T, frameskip=1, frameheight=1, path=None, filename='tmp',preprocessor=preprocessor)

    true = eval_data.value_of_data(gamma, False)
    dic.update({'ON POLICY': [float(true), 0]})
    print('V(pi_b): ',behavior_data.value_of_data(gamma, False), 'V(pi_b) Normalized: ',behavior_data.value_of_data(gamma, True))
    print('V(pi_e): ',eval_data.value_of_data(gamma, False), 'V(pi_e) Normalized: ',eval_data.value_of_data(gamma, True))

    get_Qs = getQs(behavior_data, pi_e, processor, len(actions))

    for model in models:
        if (model == 'MBased_Approx') or (model == 'MBased_MLE'):
            if model == 'MBased_MLE':
                print('*'*20)
                print('MLE estimator not implemented for continuous state space. Using MBased_Approx instead')
                print('*'*20)
            MBased_max_trajectory_length = 50 if not debug else 1
            batchsize = 32
            mbased_num_epochs = 100
            MDPModel = ApproxModel(gamma, None, MBased_max_trajectory_length, FRAMESKIP, FRAMEHEIGHT)
            mdpmodel = MDPModel.run(env, behavior_data, mbased_num_epochs, batchsize, modeltype)

            Qs_model_based = get_Qs.get(mdpmodel)
            out = estimate(Qs_model_based, behavior_data, gamma,'MBased_Approx', true)
            dic.update(out)

        elif model == 'MFree_Reg':
            DMRegression = DirectMethodRegression(behavior_data, gamma, FRAMESKIP, FRAMEHEIGHT, modeltype)
            dm_max_epochs = 80 if not debug else 1
            if modeltype == 'linear':
                dm_model_Q = DMRegression.run_linear(env, pi_b, pi_e, dm_max_epochs, epsilon=0.001)
            else:
                _,dm_model_Q = DMRegression.run_NN(env, pi_b, pi_e, dm_max_epochs, epsilon=0.001)

            dm_model = QWrapper(dm_model_Q, None, is_model=True, action_space_dim=env.n_actions, modeltype=modeltype)
            Qs_DM_based = get_Qs.get(dm_model)

            out = estimate(Qs_DM_based, behavior_data, gamma,'DM Regression', true)
            dic.update(out)
        elif model == 'MFree_FQE':
            FQE = FittedQEvaluation(behavior_data, gamma, FRAMESKIP, FRAMEHEIGHT, modeltype)

            fqe_max_epochs = 160 if not debug else 1
            if modeltype == 'linear':
                fqe_Q = FQE.run_linear(env, pi_b, pi_e, fqe_max_epochs, epsilon=0.001)
            else:
                _,_,fqe_Q = FQE.run_NN(env, pi_b, pi_e, fqe_max_epochs, epsilon=0.001)

            fqe_model = QWrapper(fqe_Q, None, is_model=True, action_space_dim=env.n_actions, modeltype=modeltype)
            Qs_FQE_based = get_Qs.get(fqe_model)

            out = estimate(Qs_FQE_based, behavior_data, gamma, 'FQE', true)
            dic.update(out)
        elif model == 'MFree_IH':
            ih_max_epochs = 10001 if not debug else 1
            ih_matrix_size = 1024
            inf_horizon = IH(behavior_data, 30, 1e-3, 3e-3, gamma, False, modeltype)
            inf_hor_output = inf_horizon.evaluate(env, ih_max_epochs, ih_matrix_size)
            inf_hor_output /= 1/np.sum(gamma ** np.arange(max(behavior_data.lengths())))
            dic.update({'IH': [inf_hor_output, (inf_hor_output - true )**2]})
        elif model == 'MFree_MRDR':
            mrdr = MRDR(behavior_data, gamma, FRAMESKIP, FRAMEHEIGHT, modeltype)

            mrdr_max_epochs = 80 if not debug else 1
            mrdr_matrix_size = 1024

            if modeltype == 'linear':
                mrdr_Q = mrdr.run(pi_e)
            else:
                _,_,mrdr_Q = mrdr.run_NN(env, pi_b, pi_e, mrdr_max_epochs, mrdr_matrix_size, epsilon=0.001)

            mrdr_model = QWrapper(mrdr_Q, None, is_model=True, action_space_dim=env.n_actions, modeltype=modeltype)
            Qs_mrdr_based = get_Qs.get(mrdr_model)

            out = estimate(Qs_mrdr_based, behavior_data, gamma, 'MRDR', true)
            dic.update(out)
        elif model == 'MFree_Retrace_L':
            # print('*'*20)
            # print('Retrace(lambda) estimator not implemented for continuous state space')
            # print('*'*20)
            retrace = Retrace(behavior_data, gamma, FRAMESKIP, FRAMEHEIGHT, modeltype, lamb=.9)

            retrace_max_epochs = 80 if not debug else 1
            _,_,retrace_Q = retrace.run_NN(env, pi_b, pi_e, retrace_max_epochs, 'retrace', epsilon=0.001)
            retrace_model = QWrapper(retrace_Q, None, is_model=True, action_space_dim=env.n_actions, modeltype='mlp') # use mlp-based wrapper even for linear
            Qs_retrace_based = get_Qs.get(retrace_model)
            out = estimate(Qs_retrace_based, behavior_data, gamma, 'Retrace(lambda)', true)
            dic.update(out)

            _,_,tree_Q = retrace.run_NN(env, pi_b, pi_e, retrace_max_epochs, 'tree-backup', epsilon=0.001)
            tree_model = QWrapper(tree_Q, None, is_model=True, action_space_dim=env.n_actions, modeltype='mlp')
            Qs_tree_based = get_Qs.get(tree_model)
            out = estimate(Qs_tree_based, behavior_data, gamma, 'Tree-Backup', true)
            dic.update(out)

            _,_,q_lambda_Q = retrace.run_NN(env, pi_b, pi_e, retrace_max_epochs, 'Q^pi(lambda)', epsilon=0.001)
            q_lambda_model = QWrapper(q_lambda_Q, None, is_model=True, action_space_dim=env.n_actions, modeltype='mlp')
            Qs_q_lambda_based = get_Qs.get(q_lambda_model)
            out = estimate(Qs_q_lambda_based, behavior_data, gamma, 'Q^pi(lambda)', true)
            dic.update(out)


        elif model == 'IS':
            out = estimate([], behavior_data, gamma, 'IS', true, True)
            dic.update(out)
        else:
            print(model, ' is not a valid method')

        analysis(dic)

    return analysis(dic)

def main():
    debug = False
    parser = argparse.ArgumentParser(description='Distribute experiments across ec2 instances.')

    subparsers = parser.add_subparsers(dest='subcommand', help='Local or Distributed (AWS)')
    # parser.add_argument("-v", ...)

    a_parser = subparsers.add_parser("local")
    a_parser.add_argument('cfg', help='config file', type=str)
    a_parser.add_argument('-m', '--models', help='which models to use', type=str, nargs='+', required=True)


    b_parser = subparsers.add_parser("distributed")
    b_parser.add_argument('env', help='Environment type: toy_model, mc, mc_pixel,...', type=str)
    b_parser.add_argument('gamma', help='gamma', type=float)
    b_parser.add_argument('horizon', help='horizon', type=int)
    b_parser.add_argument('base_policy', help='base_policy', type=int)
    b_parser.add_argument('eval_policy', help='eval_policy', type=int)
    b_parser.add_argument('stochastic_env', help='stochastic_env', type=int)
    b_parser.add_argument('stochastic_rewards', help='stochastic_rewards', type=int)
    b_parser.add_argument('sparse_rewards', help='sparse_rewards', type=int)
    b_parser.add_argument('num_traj', help='num_traj', type=int)
    b_parser.add_argument('is_pomdp', help='is_pomdp', type=int)
    b_parser.add_argument('pomdp_horizon', help='pomdp_horizon', type=int)
    b_parser.add_argument('seed', help='seed', type=int)
    b_parser.add_argument('modeltype', help='Model type: linear, mlp, conv', type=str)
    b_parser.add_argument('to_regress_pi_b', help='Regress on pib instead of using pib exactly', type=int)

    b_parser.add_argument('experiment_number', help='experiment_number', type=int)
    b_parser.add_argument('access', help='access', type=str)
    b_parser.add_argument('secret', help='secret', type=str)


    # a_parser.add_argument("something", choices=['a1', 'a2'])
    args = parser.parse_args()

    if args.subcommand == 'local':
        with open('cfgs/{0}'.format(args.cfg), 'r') as f:
            hyperparam = json.load(f)

        hyperparam['subcommand'] = 'local'
        models = args.models
        if models is None:
            print('*'*20)
            print('ERROR: Please supply models (space separated)')
            print('*'*20)
            sys.exit(0)

        valid_models = {'MBased_MLE':1,'MBased_Approx':1, 'MFree_Reg':1, 'MFree_FQE':1, 'MFree_IH':1, 'MFree_MRDR':1, 'IS':1, 'MFree_Retrace_L':1}
        for model in models:
            assert model in valid_models, '{0} is not valid. Please check your arguments'.format(model)

    else:
        hyperparam = vars(args)
        models = ['MFree_Retrace_L', 'MFree_MRDR', 'MFree_IH', 'MFree_FQE', 'MBased_MLE', 'MFree_Reg', 'IS'] # all

    if hyperparam['subcommand'] == 'distributed':
        from pyvirtualdisplay import Display
        display = Display(visible=0, size=(1280, 1024))
        display.start()

    if hyperparam['env'] == 'toy_graph':
        result = toy_graph(hyperparam, models, debug)
    elif hyperparam['env'] == 'toy_mc':
        if any(['MBased_MLE' in x for x in models]):
            print('Warning: Model Based methods will take a very long time to rollout')
        result = toy_mc(hyperparam, models, debug)
    elif hyperparam['env'] == 'mc':
        if any(['MBased_MLE' in x for x in models]):
            print('Warning: Model Based methods will take a very long time to rollout')
        result = mc(hyperparam, models, debug)
    elif hyperparam['env'] == 'pixel_mc':
        if any(['MBased_MLE' in x for x in models]):
            print('Warning: Model Based methods will take a very long time to rollout')
        result = pixel_mc(hyperparam, models, debug)
    elif hyperparam['env'] == 'enduro':
        if any(['MBased_MLE' in x for x in models]):
            print('Warning: Model Based methods will take a very long time to rollout')
        result = enduro(hyperparam, models, debug)
    elif hyperparam['env'] == 'breakout':
        if any(['MBased_MLE' in x for x in models]):
            print('Warning: Model Based methods will take a very long time to rollout')
        result = breakout(hyperparam, models, debug)
    elif hyperparam['env'] == 'baird':
        if any(['MBased_MLE' in x for x in models]):
            print('Warning: Model Based methods will take a very long time to rollout')
        result = baird(hyperparam, models, debug)
    elif hyperparam['env'] == 'gridworld':
        if any(['MBased_MLE' in x for x in models]):
            print('Warning: Model Based methods will take a very long time to rollout')
        result = gridworld(hyperparam, models, debug)
    elif hyperparam['env'] == 'pixel_gridworld':
        if any(['MBased_MLE' in x for x in models]):
            print('Warning: Model Based methods will take a very long time to rollout')
        result = pixel_gridworld(hyperparam, models, debug)
    else:
        raise NotImplemented

    if hyperparam['subcommand'] == 'distributed':
        aws(hyperparam, result)

def aws(hyperparam, result):
    session = boto3.Session(aws_access_key_id=hyperparam['access'], aws_secret_access_key=hyperparam['secret'], region_name='us-east-2')
    s3 = session.resource('s3')
    upload_files_to_s3(s3, '%s-ope-experiments' % '-'.join(hyperparam['env'].split('_')), hyperparam['experiment_number'], result)

def upload_files_to_s3(s3, bucket_name, experiment_number, result):
    '''
    Uploads files to s3 bucket_name
    '''

    bucket = s3.Bucket(bucket_name)

    with open('results.json', 'w') as outfile:
        json.dump(dict(result), outfile)

    bucket.upload_file('results.json', '/'.join(['exp_%05d' % experiment_number,'results.json']) )
    return True

if __name__ == '__main__':
    # Local:
    # python paper.py local base_toy_graph_cfg.json -m MFree_Retrace_L MFree_MRDR MFree_IH MFree_FQE MBased_MLE MFree_Reg IS
    # python paper.py local base_pixel_mc_cfg.json -m IS
    # python paper.py local base_pixel_gridworld_cfg.json -m IS

    # AWS:
    # python paper.py distributed toy_graph .98 4 4 1 0 0 1 512 0 2 10000 conv 0 0 0
    # python paper.py distributed toy_mc .98 250 40 60 0 0 0 512 0 2 10000 conv 0 0 0
    # python paper.py distributed pixel_gridworld .96 25 100 10 1 1 1 10 0 2 10000 conv 0 0 0
    # python3 paper.py distributed enduro .999 25 1 0 0 0 0 10 0 2 10000 conv 0 0 0 0

    # ffmpeg -i ./videos/enduro/pi_b_%05d.jpg ./videos/enduro/pi_b_movie.mp4
    main()




