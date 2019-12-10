import numpy as np
import tensorflow as tf
from time import sleep
import sys
import os
from tqdm import tqdm
from tensorflow.python import debug as tf_debug
import json

from scipy.optimize import linprog
from scipy.optimize import minimize
import quadprog

import keras
from keras.layers     import Dense, Conv2D, Flatten, MaxPool2D, concatenate, UpSampling2D, Reshape, Lambda, Conv2DTranspose
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras import backend as K
from tqdm import tqdm

from ope.utls.thread_safe import threadsafe_generator
from keras import regularizers

# Hyper parameter
#Learning_rate = 1e-3
#initial_stddev = 0.5

#Training Parameter
training_batch_size = 1024 #1024 * 2**2
training_maximum_iteration = 40001
TEST_NUM = 0
NUMBER_OF_REPEATS = 1

class InfiniteHorizonOPE(object):
    def __init__(self, data, w_hidden, Learning_rate, reg_weight, gamma, discrete, modeltype, env=None, processor=None):

        self.data = data
        self.modeltype = modeltype
        self.gamma = gamma
        self.is_discrete = discrete
        self.processor = processor
        if self.is_discrete:
            self.obs_dim = env.num_states() if env is not None else self.data.num_states()
            self.den_discrete = Density_Ratio_discounted(self.obs_dim, gamma)
        else:
            # self.g = tf.Graph()
            # with self.g.as_default():
            #     with tf.variable_scope('infhorizon', reuse=False):
            #         self._build_graph(w_hidden, Learning_rate, reg_weight)
            #         self._init_session()
            pass

    def build_model(self, input_size, scope, action_space_dim=3, modeltype='conv'):
        isStart = keras.layers.Input(shape=(1,), name='dummy')
        state =  keras.layers.Input(shape=input_size, name='state')
        next_state =  keras.layers.Input(shape=input_size, name='next_state')
        median_dist = keras.layers.Input(shape=(1,), name='med_dist')
        policy_ratio = keras.layers.Input(shape=(1,), name='policy_ratio')

        if modeltype == 'conv':
            def init(): return keras.initializers.RandomNormal(mean=0.0, stddev=.003, seed=np.random.randint(2**32))
            conv1 = Conv2D(8, (7,7), strides=(3,3), padding='same', data_format='channels_first', activation='elu',kernel_initializer=init(), bias_initializer=init())
            pool1 = MaxPool2D(data_format='channels_first')
            conv2 = Conv2D(16, (3,3), strides=(1,1), padding='same', data_format='channels_first', activation='elu',kernel_initializer=init(), bias_initializer=init())
            pool2 = MaxPool2D(data_format='channels_first')
            flat1 = Flatten(name='flattened')
            out = Dense(1, activation='linear',kernel_initializer=init(), bias_initializer=init())
            output = Lambda(lambda x: tf.exp(tf.clip_by_value(x,-10,10)))

            w = output(out(flat1(pool2(conv2(pool1(conv1(state)))))))
            w_next = output(out(flat1(pool2(conv2(pool1(conv1(next_state)))))))

            trainable_model = keras.models.Model(inputs=[state,next_state,policy_ratio,isStart,median_dist], outputs=[w])
            w_model = keras.models.Model(inputs=[state], outputs=w)
        elif modeltype == 'conv1':
            def init(): return keras.initializers.RandomNormal(mean=0.0, stddev=.003, seed=np.random.randint(2**32))
            conv1 = Conv2D(8, (2,2), strides=(1,1), padding='same', data_format='channels_first', activation='elu',kernel_initializer=init(), bias_initializer=init())
            pool1 = MaxPool2D(data_format='channels_first')
            conv2 = Conv2D(16, (2,2), strides=(1,1), padding='same', data_format='channels_first', activation='elu',kernel_initializer=init(), bias_initializer=init())
            pool2 = MaxPool2D(data_format='channels_first')
            flat1 = Flatten(name='flattened')
            out = Dense(1, activation='linear',kernel_initializer=init(), bias_initializer=init())
            output = Lambda(lambda x: tf.exp(tf.clip_by_value(x,-10,10)))

            w = output(out(flat1(pool2(conv2(pool1(conv1(state)))))))
            w_next = output(out(flat1(pool2(conv2(pool1(conv1(next_state)))))))


            trainable_model = keras.models.Model(inputs=[state,next_state,policy_ratio,isStart,median_dist], outputs=[w])
            w_model = keras.models.Model(inputs=[state], outputs=w)
        elif modeltype == 'linear':
            def init(): return keras.initializers.RandomNormal(mean=0.0, stddev=.003, seed=np.random.randint(2**32))
            dense1 = Dense(1, activation='linear', name='out',kernel_initializer=init(), bias_initializer=keras.initializers.Zeros())
            output = Lambda(lambda x: tf.exp(tf.clip_by_value(x,-10,10)))

            w = output(dense1(state))
            w_next = output(dense1(next_state))

            trainable_model = keras.models.Model(inputs=[state,next_state,policy_ratio,isStart,median_dist], outputs=[w])
            w_model = keras.models.Model(inputs=[state], outputs=w)
        else:
            def init(): return keras.initializers.RandomNormal(mean=0.0, stddev=.003, seed=np.random.randint(2**32))
            dense1 = Dense(16, activation='relu',kernel_initializer=init(), bias_initializer=keras.initializers.Zeros())
            dense2 = Dense(8, activation='relu',kernel_initializer=init(), bias_initializer=keras.initializers.Zeros())
            dense3 = Dense(1, activation='linear', name='out',kernel_initializer=init(), bias_initializer=keras.initializers.Zeros())
            output = Lambda(lambda x: tf.exp(tf.clip_by_value(x,-10,10)))

            w = output(dense3(dense2(dense1(state))))
            w_next = output(dense3(dense2(dense1(next_state))))


            trainable_model = keras.models.Model(inputs=[state,next_state,policy_ratio,isStart,median_dist], outputs=[w])
            w_model = keras.models.Model(inputs=[state], outputs=w)

        # rmsprop = keras.optimizers.RMSprop(lr=0.001, rho=0.95, epsilon=1e-08, decay=1e-3)#, clipnorm=1.)
        adam = keras.optimizers.Adam()

        trainable_model.add_loss(self.IH_loss(next_state,w,w_next,policy_ratio,isStart, median_dist, self.modeltype))
        trainable_model.compile(loss=None, optimizer=adam, metrics=['accuracy'])
        return trainable_model, w_model

    @staticmethod
    def IH_loss(next_state, w, w_next,policy_ratio,isStart, med_dist, modeltype):
        # change from tf to K.backend?
        norm_w = tf.reduce_mean(w)

        # calculate loss function
        x = (1-isStart) * w * policy_ratio + isStart * norm_w - w_next
        x = tf.reshape(x,[-1,1])

        diff_xx = tf.expand_dims(next_state, 0) - tf.expand_dims(next_state, 1)
        if modeltype in ['conv', 'conv1']:
            K_xx = tf.exp(-tf.reduce_sum(tf.square(diff_xx),axis=[-1, -2, -3])/(2.0*med_dist*med_dist))#*med_dist))
        else:
            K_xx = tf.exp(-tf.reduce_sum(tf.square(diff_xx),axis=[-1])/(2.0*med_dist*med_dist))#*med_dist))

        loss_xx = tf.matmul(tf.matmul(tf.transpose(x),K_xx),x)#/(n_x*n_x)

        loss = tf.squeeze(loss_xx)/(norm_w*norm_w)
        return tf.reduce_mean(loss)


    def run_NN(self, env, max_epochs, batch_size, epsilon=0.001, modeltype_overwrite =None):

        self.dim_of_actions = env.n_actions
        self.Q_k = None

        # earlyStopping = EarlyStopping(monitor='val_loss', min_delta=1e-4,  patience=10, verbose=1, mode='min', restore_best_weights=True)
        # reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, min_delta=1e-4, mode='min')

        self.more_callbacks = [] #[earlyStopping, reduce_lr_loss]

        im = self.data.states()[0]
        if self.processor: im = self.processor(im)
        if self.modeltype in ['conv', 'conv1']:
            trainable_model, state_to_w = self.build_model(im.shape[1:], 'w', modeltype=modeltype_overwrite if modeltype_overwrite is not None else self.modeltype)
            state_to_w.predict([im])
            self.state_to_w = state_to_w
            self.trainable_model = trainable_model
        else:
            trainable_model, state_to_w = self.build_model((np.prod(im.shape[1:]),), 'w', modeltype=self.modeltype)
            state_to_w.predict([[im[0].reshape(-1)]])
            self.state_to_w = state_to_w
            self.trainable_model = trainable_model
        values = []

        print('Training: IH')
        losses = []
        for k in tqdm(range(1)):

            dataset_length = self.data.num_tuples()
            perm = np.random.permutation(range(dataset_length))
            eighty_percent_of_set = int(1.*len(perm))
            training_idxs = perm[:eighty_percent_of_set]
            validation_idxs = perm[eighty_percent_of_set:]
            training_steps_per_epoch = 1 #int(1. * np.ceil(len(training_idxs)/float(batch_size)))
            validation_steps_per_epoch = int(np.ceil(len(validation_idxs)/float(batch_size)))
            train_gen = self.generator(env, training_idxs, fixed_permutation=True, batch_size=batch_size)
            # val_gen = self.generator(env, validation_idxs, fixed_permutation=True, batch_size=batch_size)

            M = 5
            hist = self.trainable_model.fit_generator(train_gen,
                               steps_per_epoch=training_steps_per_epoch,
                               # validation_data=val_gen,
                               # validation_steps=validation_steps_per_epoch,
                               # epochs=max_epochs,
                               # max_queue_size=50,
                               # workers=1,
                               # use_multiprocessing=False,
                               # verbose=1,
                               callbacks = self.more_callbacks)
        return state_to_w

        # return np.mean(values[-10:]), self.Q_k, self.Q_k_all

    def euclidean(self, X, Y):
        distance = np.zeros((len(X), len(Y)))
        for row,x in enumerate(X):
            for col,y in enumerate(Y):
                y_row,y_col = np.unravel_index(np.argmin(y.reshape(-1), y.shape))
                x_row,x_col = np.unravel_index(np.argmin(y.reshape(-1), y.shape))
                distance = np.sqrt((x_row-y_row)**2 + (x_col+y_col)**2)
        return distance

    @threadsafe_generator
    def generator(self, env, all_idxs, fixed_permutation=False,  batch_size = 64):
        data_length = len(all_idxs)
        steps = int(np.ceil(data_length/float(batch_size)))

        n = len(self.data)
        T = max(self.data.lengths())
        n_dim = self.data.n_dim
        n_actions = self.data.n_actions


        S = np.hstack([self.data.states()[:,[0]], self.data.states()])
        SN = np.hstack([self.data.states()[:,[0]], self.data.next_states()])
        PI0 = np.hstack([self.data.base_propensity()[:,[0]], self.data.base_propensity()])
        PI1 = np.hstack([self.data.target_propensity()[:,[0]], self.data.target_propensity()])

        ACTS = np.hstack([np.zeros_like(self.data.actions()[:,[0]]), self.data.actions()])
        pi0 = []
        pi1 = []
        for i in range(len(ACTS)):
            pi0_ = []
            pi1_ = []
            for j in range(len(ACTS[1])):
                a = ACTS[i,j]
                pi0_.append(PI0[i,j,a])
                pi1_.append(PI1[i,j,a])
            pi0.append(pi0_)
            pi1.append(pi1_)

        PI0 = np.array(pi0)
        PI1 = np.array(pi1)


        REW = np.hstack([np.zeros_like(self.data.rewards()[:,[0]]), self.data.rewards()])
        ISSTART = np.zeros_like(REW)
        ISSTART[:,0] = 1.

        PROBS = np.repeat(np.atleast_2d(self.gamma**np.arange(-1,REW.shape[1]-1)), REW.shape[0], axis=0).reshape(REW.shape)

        S = np.vstack(S)
        SN = np.vstack(SN)
        PI1 = PI1.reshape(-1)
        PI0 = PI0.reshape(-1)
        ISSTART = ISSTART.reshape(-1)
        PROBS = PROBS.reshape(-1)
        PROBS /= sum(PROBS)

        N = S.shape[0]

        subsamples = np.random.choice(N, len(S))
        bs = batch_size
        num_batches = max(len(subsamples) // bs,1)
        med_dist = []
        for batch_num in tqdm(range(num_batches)):
            low_ = batch_num * bs
            high_ = (batch_num + 1) * bs
            sub = subsamples[low_:high_]
            if self.modeltype in ['conv']:
                s = self.processor(S[sub])
            else:
                s = S[sub].reshape(len(sub),-1)[...,None,None]

            med_dist.append(np.sum(np.square(s[None, :, :] - s[:, None, :]), axis = tuple([-3,-2,-1])))

        med_dist = np.sqrt(np.median(np.array(med_dist).reshape(-1)[np.array(med_dist).reshape(-1) > 0]))

        while True:
            # perm = np.random.permutation(all_idxs)
            for batch in np.arange(steps):
                # batch_idxs = perm[(batch*batch_size):((batch+1)*batch_size)]
                batch_idxs = np.random.choice(S.shape[0], batch_size, p=PROBS)

                if self.modeltype in ['conv', 'conv1']:
                    state = self.processor(S[batch_idxs])
                    next_state = self.processor(SN[batch_idxs])
                else:
                    state = S[batch_idxs].reshape(len(batch_idxs),-1)#[...,None,None]
                    next_state = SN[batch_idxs].reshape(len(batch_idxs),-1)#[...,None,None]

                policy_ratio = PI1[batch_idxs] / PI0[batch_idxs]
                isStart = ISSTART[batch_idxs]
                median_dist = np.repeat(med_dist, batch_size)

                yield ([state,next_state,policy_ratio,isStart,median_dist], [])


    def evaluate(self, env, max_epochs, matrix_size):
        dataset = self.data
        if self.is_discrete:

            S = np.squeeze(dataset.states())
            SN = np.squeeze(dataset.next_states())
            PI0 = dataset.base_propensity()
            PI1 = dataset.target_propensity()
            REW = dataset.rewards()
            ACTS = dataset.actions()

            for episode in range(len(S)):
                discounted_t = 1.0
                initial_state = S[episode][0]
                for (s,a,sn,r,pi1,pi0) in zip(S[episode],ACTS[episode],SN[episode], REW[episode], PI1[episode], PI0[episode]):
                    discounted_t *= self.gamma
                    policy_ratio = (pi1/pi0)[a]
                    self.den_discrete.feed_data(s, sn, initial_state, policy_ratio, discounted_t)
                self.den_discrete.feed_data(-1, initial_state, initial_state, 1, 1-discounted_t)


            x, w = self.den_discrete.density_ratio_estimate()
            total_reward = 0.0
            self_normalizer = 0.0
            for episode in range(len(S)):
                discounted_t = 1.0
                for (s,a,sn,r,pi1,pi0) in zip(S[episode],ACTS[episode],SN[episode], REW[episode], PI1[episode], PI0[episode]):
                    policy_ratio = (pi1/pi0)[a]
                    total_reward += w[s] * policy_ratio * r * discounted_t
                    self_normalizer += w[s] * policy_ratio * discounted_t
                    discounted_t *= self.gamma

            return total_reward / self_normalizer

        else:

            batch_size = matrix_size
            # Here Linear = linear NN
            self.state_to_w = self.run_NN(env, max_epochs, batch_size, epsilon=0.001)


            S = self.data.states() #np.hstack([self.data.states()[:,[0]], self.data.states()])
            PI0 = self.data.base_propensity() #np.hstack([self.data.base_propensity()[:,[0]], self.data.base_propensity()])
            PI1 = self.data.target_propensity() #np.hstack([self.data.target_propensity()[:,[0]], self.data.target_propensity()])

            ACTS = self.data.actions() #np.hstack([np.zeros_like(self.data.actions()[:,[0]]), self.data.actions()])
            pi0 = []
            pi1 = []
            for i in range(len(ACTS)):
                pi0_ = []
                pi1_ = []
                for j in range(len(ACTS[1])):
                    a = ACTS[i,j]
                    pi0_.append(PI0[i,j,a])
                    pi1_.append(PI1[i,j,a])
                pi0.append(pi0_)
                pi1.append(pi1_)

            PI0 = np.array(pi0)
            PI1 = np.array(pi1)

            REW = self.data.rewards() #np.hstack([np.zeros_like(self.data.rewards()[:,[0]]), self.data.rewards()])

            PROBS = np.repeat(np.atleast_2d(self.gamma**np.arange(REW.shape[1])), REW.shape[0], axis=0).reshape(REW.shape)

            S = np.vstack(S)
            PI1 = PI1.reshape(-1)
            PI0 = PI0.reshape(-1)
            PROBS = PROBS.reshape(-1)
            REW = REW.reshape(-1)

            predict_batch_size = max(128, batch_size)
            steps = int(np.ceil(S.shape[0]/float(predict_batch_size)))
            densities = []
            for batch in np.arange(steps):
                batch_idxs = np.arange(S.shape[0])[(batch*predict_batch_size):((batch+1)*predict_batch_size)]

                if self.modeltype in ['conv', 'conv1']:
                    s = self.processor(S[batch_idxs])
                    densities.append(self.state_to_w.predict(s))
                else:
                    s = S[batch_idxs]
                    s = s.reshape(s.shape[0], -1)
                    densities.append(self.state_to_w.predict(s))

            densities = np.vstack(densities).reshape(-1)
            return self.off_policy_estimator_density_ratio(REW, PROBS, PI1/PI0, densities)

    @staticmethod
    def off_policy_estimator_density_ratio(rew, prob, ratio, den_r):
        return np.sum(prob * den_r * ratio * rew)/np.sum(prob * den_r * ratio)

    def get_model_params(self):
        # get trainable params.
        model_names = []
        model_params = []
        model_shapes = []
        with self.g.as_default():
            t_vars = tf.trainable_variables()
            for var in t_vars:
                if var.name.startswith('infhorizon'):
                    param_name = var.name
                    p = self.sess.run(var)
                    model_names.append(param_name)
                    params = np.round(p*10000).astype(np.int).tolist()
                    model_params.append(params)
                    model_shapes.append(p.shape)
        return model_params, model_shapes, model_names
    def set_model_params(self, params):
        with self.g.as_default():
            t_vars = tf.trainable_variables()
            idx = 0
            for var in t_vars:
                if var.name.startswith('infhorizon'):
                    pshape = tuple(var.get_shape().as_list())
                    p = np.array(params[idx])
                    assert pshape == p.shape, "inconsistent shape"
                    assign_op, pl = self.assign_ops[var]
                    self.sess.run(assign_op, feed_dict={pl.name: p/10000.})
                    idx += 1
    def load_json(self, jsonfile='infhorizon.json'):
        with open(jsonfile, 'r') as f:
            params = json.load(f)
        self.set_model_params(params)
    def save_json(self, jsonfile='infhorizon.json'):
        model_params, model_shapes, model_names = self.get_model_params()
        qparams = []
        for p in model_params:
            qparams.append(p)
        with open(jsonfile, 'wt') as outfile:
            json.dump(qparams, outfile, sort_keys=True, indent=0, separators=(',', ': '))


def linear_solver(n, M):
    M -= np.amin(M) # Let zero sum game at least with nonnegative payoff
    c = np.ones((n))
    b = np.ones((n))
    res = linprog(-c, A_ub = M.T, b_ub = b)
    w = res.x
    return w/np.sum(w)

def quadratic_solver(n, M, regularizer):
    qp_G = np.matmul(M, M.T)
    qp_G += regularizer * np.eye(n)

    qp_a = np.zeros(n, dtype = np.float64)

    qp_C = np.zeros((n,n+1), dtype = np.float64)
    for i in range(n):
        qp_C[i,0] = 1.0
        qp_C[i,i+1] = 1.0
    qp_b = np.zeros(n+1, dtype = np.float64)
    qp_b[0] = 1.0
    meq = 1
    res = quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)
    w = res[0]
    return w

class Density_Ratio_discounted(object):
    def __init__(self, num_state, gamma):
        self.num_state = num_state
        self.Ghat = np.zeros([num_state, num_state], dtype = np.float64)
        self.Nstate = np.zeros([num_state, 1], dtype = np.float64)
        self.initial_b = np.zeros([num_state], dtype = np.float64)
        self.gamma = gamma

    def reset(self):
        num_state = self.num_state
        self.Ghat = np.zeros([num_state, num_state], dtype = np.float64)
        self.Nstate = np.zeros([num_state, 1], dtype = np.float64)

    def feed_data(self, cur, next, initial, policy_ratio, discounted_t):
        if cur == -1:
            self.Ghat[next, next] -= discounted_t
        else:
            self.Ghat[cur, next] += policy_ratio * discounted_t
            self.Ghat[cur, initial] += (1-self.gamma)/self.gamma * discounted_t
            self.Ghat[next, next] -= discounted_t
            self.Nstate[cur] += discounted_t

    def density_ratio_estimate(self, regularizer = 0.001):
        Frequency = self.Nstate.reshape(-1)
        tvalid = np.where(Frequency >= 1e-20)
        G = np.zeros_like(self.Ghat)
        Frequency = Frequency/np.sum(Frequency)
        G[tvalid] = self.Ghat[tvalid]/(Frequency[:,None])[tvalid]
        n = self.num_state
        x = quadratic_solver(n, G/50.0, regularizer)
        w = np.zeros(self.num_state)
        w[tvalid] = x[tvalid]/Frequency[tvalid]
        return x, w



