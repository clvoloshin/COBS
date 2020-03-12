import numpy as np
import scipy.signal as signal
import keras
from keras.models     import Sequential
from keras.layers     import Dense, Conv2D, Flatten, MaxPool2D, concatenate, UpSampling2D, Reshape, Lambda, Conv2DTranspose
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from ope.utls.thread_safe import threadsafe_generator
import os
from keras import regularizers
import time
from copy import deepcopy
from sklearn.linear_model import LinearRegression, LogisticRegression

class ApproxModel(object):
    def __init__(self, gamma, filename, max_traj_length=None, frameskip=2, frameheight=2, processor=None, action_space_dim=3):
        self.gamma = gamma
        self.filename = filename
        self.override_done = True if max_traj_length is not None else False
        self.max_traj_length = 200 if (max_traj_length is None) else max_traj_length
        self.frameskip = frameskip
        self.frameheight = frameheight
        self.action_space_dim = action_space_dim
        self.processor = processor

    @staticmethod
    def sample(transitions, N):
        idxs = np.random.choice(np.arange(len(transitions)), size=N)
        return transitions[idxs]

    def create_T_model(self, input_size):

        inp = keras.layers.Input(input_size, name='frames')
        actions = keras.layers.Input((self.action_space_dim,), name='mask')
        def init(): return keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001, seed=np.random.randint(2**32))
        # conv1 = Conv2D(32, kernel_size=16, strides=1, activation='elu', data_format='channels_first', padding='same')(inp)
        # pool1 = MaxPool2D((2,2), data_format='channels_first')(conv1)
        # conv2 = Conv2D(64, kernel_size=8, strides=1, activation='elu', data_format='channels_first', padding='same')(pool1)
        # pool2 = MaxPool2D((2,2), data_format='channels_first')(conv2)
        # conv3 = Conv2D(128, kernel_size=4, strides=1, activation='elu', data_format='channels_first', padding='same')(pool2)
        # # # pool3 = MaxPool2D((2,2), data_format='channels_first')(conv3)

        # # # flat = Flatten()(pool3)
        # # # concat = concatenate([flat, actions], axis = -1)
        # # # dense1 = Dense(10, activation='relu')(concat)
        # # # dense2 = Dense(20, activation='relu')(dense1)
        # # # dense3 = Dense(int(np.prod(pool3.shape[1:])), activation='relu')(dense2)
        # # # unflatten = Reshape((int(x) for x in pool3.shape[1:]))(dense3)

        # conv4 = Conv2D(128, kernel_size=4, strides=1, activation='elu', data_format='channels_first', padding='same')(conv3)
        # up1 = UpSampling2D((2,2), data_format='channels_first')(conv4)
        # conv5 = Conv2D(64, kernel_size=8, strides=1, activation='elu', data_format='channels_first', padding='same')(up1)
        # up2 = UpSampling2D((2,2), data_format='channels_first')(conv5)
        # out = Conv2D(self.action_space_dim, kernel_size=16, strides=1, activation='linear', data_format='channels_first', padding='same', name='T')(up2)

        conv1 = Conv2D(8, (7,7), strides=(4,4), padding='same', data_format='channels_first', activation='elu',kernel_initializer=init(), bias_initializer=init(), kernel_regularizer=regularizers.l2(1e-6))(inp)
        pool1 = MaxPool2D(data_format='channels_first')(conv1)
        conv2 = Conv2D(16, (3,3), strides=(1,1), padding='same', data_format='channels_first', activation='elu',kernel_initializer=init(), bias_initializer=init(), kernel_regularizer=regularizers.l2(1e-6))(pool1)


        conv3 = Conv2DTranspose(16, (3, 3), strides=(1, 1), padding='same', data_format='channels_first', activation='elu',kernel_initializer=init(), bias_initializer=init(), kernel_regularizer=regularizers.l2(1e-6))(conv2)
        up1 = UpSampling2D(data_format='channels_first')(conv3)
        out = Conv2DTranspose(self.action_space_dim, (7,7), strides=(4,4), padding='same', data_format='channels_first', activation='elu',kernel_initializer=init(), bias_initializer=init(), kernel_regularizer=regularizers.l2(1e-6), name='T')(up1)

        def filter_out(out):
            filtered_output = tf.boolean_mask(out, actions, axis = 0)
            filtered_output = K.expand_dims(filtered_output, axis=1)
            return filtered_output

        filtered_output = Lambda(filter_out)(out)

        # model = keras.models.Model(input=[inp, actions], output=[out])
        model = keras.models.Model(input=[inp, actions], output=[filtered_output])

        all_T = keras.models.Model(inputs=[inp],
                                 outputs=model.get_layer('T').output)


        rmsprop = keras.optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0, clipnorm=1.)
        model.compile(loss='mse', optimizer=rmsprop)

        return model, all_T


    def create_full_model(self, input_size):

        inp = keras.layers.Input(input_size, name='frames')
        actions = keras.layers.Input((self.action_space_dim,), name='mask')
        def init(): return keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001, seed=np.random.randint(2**32))

        if self.modeltype == 'conv':
            # Compress
            conv1 = Conv2D(8, (7,7), strides=(2,2), padding='same', data_format='channels_first', activation='elu',kernel_initializer=init(), bias_initializer=init(), kernel_regularizer=regularizers.l2(1e-6))(inp)
            conv2 = Conv2D(16, (5,5), strides=(2,2), padding='same', data_format='channels_first', activation='elu',kernel_initializer=init(), bias_initializer=init(), kernel_regularizer=regularizers.l2(1e-6))(conv1)
            conv3 = Conv2D(32, (3,3), strides=(1,1), padding='same', data_format='channels_first', activation='elu',kernel_initializer=init(), bias_initializer=init(), kernel_regularizer=regularizers.l2(1e-6))(conv2)

            flat = Flatten()(conv3)

            # Transition
            conv4 = Conv2DTranspose(32, (3, 3), strides=(1, 1), padding='same', data_format='channels_first', activation='elu',kernel_initializer=init(), bias_initializer=init(), kernel_regularizer=regularizers.l2(1e-6))(conv3)
            # up1 = UpSampling2D(data_format='channels_first')(conv4)
            conv5 = Conv2DTranspose(16, (5,5), strides=(2,2), padding='same', data_format='channels_first', activation='elu',kernel_initializer=init(), bias_initializer=init(), kernel_regularizer=regularizers.l2(1e-6))(conv4)
            out_T = Conv2DTranspose(self.action_space_dim, (7,7), strides=(2,2), padding='same', data_format='channels_first', activation='tanh',kernel_initializer=init(), bias_initializer=init(), kernel_regularizer=regularizers.l2(1e-6), name='all_T')(conv5)

            # Rewards
            dense1 = Dense(10, activation='relu')(flat)
            dense2 = Dense(30, activation='relu')(dense1)
            out_R = Dense(self.action_space_dim, activation='linear', name='all_R')(dense2)

            # Dones
            dense3 = Dense(10, activation='relu')(flat)
            dense4 = Dense(30, activation='relu')(dense3)
            out_D = Dense(self.action_space_dim, activation='softmax', name='all_D')(dense4)
        elif  self.modeltype == 'conv1':
            # Compress
            conv1 = Conv2D(16, (3,3), strides=(2,2), padding='same', data_format='channels_first', activation='elu',kernel_initializer=init(), bias_initializer=init(), kernel_regularizer=regularizers.l2(1e-6))(inp)
            conv2 = Conv2D(16, (2,2), strides=(1,1), padding='same', data_format='channels_first', activation='elu',kernel_initializer=init(), bias_initializer=init(), kernel_regularizer=regularizers.l2(1e-6))(conv1)
            conv3 = Conv2D(16, (2,2), strides=(1,1), padding='same', data_format='channels_first', activation='elu',kernel_initializer=init(), bias_initializer=init(), kernel_regularizer=regularizers.l2(1e-6))(conv2)

            flat = Flatten()(conv3)

            # Transition
            conv4 = Conv2DTranspose(16, (2, 2), strides=(1, 1), padding='same', data_format='channels_first', activation='elu',kernel_initializer=init(), bias_initializer=init(), kernel_regularizer=regularizers.l2(1e-6))(conv3)
            # up1 = UpSampling2D(data_format='channels_first')(conv4)
            conv5 = Conv2DTranspose(16, (2,2), strides=(1,1), padding='same', data_format='channels_first', activation='elu',kernel_initializer=init(), bias_initializer=init(), kernel_regularizer=regularizers.l2(1e-6))(conv4)
            out_T = Conv2DTranspose(self.action_space_dim, (2,2), strides=(2,2), padding='same', data_format='channels_first', activation='tanh',kernel_initializer=init(), bias_initializer=init(), kernel_regularizer=regularizers.l2(1e-6), name='all_T')(conv5)

            # Rewards
            dense1 = Dense(5, activation='relu')(flat)
            dense2 = Dense(10, activation='relu')(dense1)
            out_R = Dense(self.action_space_dim, activation='linear', name='all_R')(dense2)

            # Dones
            dense3 = Dense(5, activation='relu')(flat)
            dense4 = Dense(10, activation='relu')(dense3)
            out_D = Dense(self.action_space_dim, activation='softmax', name='all_D')(dense4)
        else:
            # Compress
            flat = Flatten()(inp)
            dense1 = Dense(128, activation='relu')(flat)
            dense2 = Dense(32, activation='relu')(dense1)
            # dense3 = Dense(128, activation='relu')(dense2)
            out = Dense(2*self.action_space_dim, activation='linear')(dense2)
            out_T = Reshape((-1,2), name='all_T')(out)


            # Rewards
            dense4 = Dense(8, activation='relu')(dense1)
            out_R = Dense(self.action_space_dim, activation='linear', name='all_R')(dense2)

            # Dones
            dense5 = Dense(8, activation='relu')(dense1)
            out_D = Dense(self.action_space_dim, activation='softmax', name='all_D')(dense4)


        def filter_out(out):
            filtered_output = tf.boolean_mask(out, actions, axis = 0)
            filtered_output = K.expand_dims(filtered_output, axis=1)
            return filtered_output

        filtered_T = Lambda(filter_out, name='T')(out_T)
        filtered_R = Lambda(filter_out, name='R')(out_R)
        filtered_D = Lambda(filter_out, name='D')(out_D)

        # model = keras.models.Model(input=[inp, actions], output=[out])
        losses = {
            "T": "mse",
            "R": "mse",
            "D": "binary_crossentropy",
        }

        model = keras.models.Model(input=[inp, actions], output=[filtered_T, filtered_R, filtered_D])

        all_model = keras.models.Model(inputs=[inp],
                                 outputs=[model.get_layer('all_T').output, model.get_layer('all_R').output, model.get_layer('all_D').output])

        # rmsprop = keras.optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0, clipnorm=1.)
        model.compile(loss=losses, optimizer='Adam')

        return model, all_model

    def create_scalar_model(self, input_size, is_R=True):

        inp = keras.layers.Input(input_size, name='frames')
        actions = keras.layers.Input((self.action_space_dim,), name='mask')

        # "The first hidden layer convolves 16 8×8 filters with stride 4 with the input image and applies a rectifier nonlinearity."
        # conv_1 = keras.layers.convolutional.Convolution2D(
        #     8, 5 , 5, subsample=(4, 4), activation='relu'
        # )(normalized)

        conv1 = Conv2D(64, kernel_size=16, strides=2, activation='relu', data_format='channels_first')(inp)
        #pool1 = MaxPool2D(data_format='channels_first')(conv1)
        conv2 = Conv2D(64, kernel_size=8, strides=2, activation='relu', data_format='channels_first')(conv1)
        #pool2 = MaxPool2D(data_format='channels_first')(conv2)
        conv3 = Conv2D(64, kernel_size=4, strides=2, activation='relu', data_format='channels_first')(conv2)
        #pool3 = MaxPool2D(data_format='channels_first')(conv3)
        flat = Flatten()(conv3)
        dense1 = Dense(10, activation='relu')(flat)
        dense2 = Dense(30, activation='relu')(dense1)
        out = Dense(self.action_space_dim, activation='sigmoid', name='all_')(dense2)
        filtered_output = keras.layers.dot([out, actions], axes=1)

        model = keras.models.Model(input=[inp, actions], output=[filtered_output])

        all_ = keras.models.Model(inputs=[inp],
                                 outputs=model.get_layer('all_').output)

        rmsprop = keras.optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0, clipnorm=1.)
        if is_R:
            model.compile(loss='mse', optimizer=rmsprop)
        else:
            model.compile(loss='binary_crossentropy', optimizer=rmsprop)

        return model, all_

    @threadsafe_generator
    def T_gen(self, env, data, batchsize=32):

        frameskip = self.frameheight #lazy
        num_batches = int(np.ceil(data.shape[0] / batchsize))
        while True:

            permutation = np.random.permutation(np.arange(len(data)))
            data = data[permutation]

            for batch_idx in np.arange(num_batches):
                low_ = batch_idx * batchsize
                high_ = (1+batch_idx) * batchsize
                batch = data[low_:high_]
                x =  batch[:,:frameskip]
                act =  batch[:, frameskip].astype(int)
                x_ = batch[:, (frameskip+1):]

                inp = env.pos_to_image(x)
                out = np.diff(env.pos_to_image(x_), axis=1)

                yield [inp, np.eye(env.n_actions)[act]], out

    @threadsafe_generator
    def D_gen(self, env, data, batchsize=32):

        frameskip = self.frameheight #lazy
        num_batches = int(np.ceil(data.shape[0] / batchsize))
        while True:

            permutation = np.random.permutation(np.arange(len(data)))
            data = data[permutation]

            for batch_idx in np.arange(num_batches):
                low_ = batch_idx * batchsize
                high_ = (1+batch_idx) * batchsize
                batch = data[low_:high_]
                x_pre =  batch[:,:frameskip]
                act =  batch[:, frameskip].astype(int)
                x__pre = batch[:, (frameskip+1):-1]
                done = batch[:, -1].astype(int)

                x = env.pos_to_image(x_pre)
                x_ = env.pos_to_image(x__pre)

                inp = np.concatenate([x, x_], axis=1)
                out = done

                yield [inp, np.eye(env.n_actions)[act]], out

    @threadsafe_generator
    def R_gen(self, env, data, batchsize=32):

        frameskip = self.frameheight #lazy
        num_batches = int(np.ceil(data.shape[0] / batchsize))
        while True:

            permutation = np.random.permutation(np.arange(len(data)))
            data = data[permutation]

            for batch_idx in np.arange(num_batches):
                low_ = batch_idx * batchsize
                high_ = (1+batch_idx) * batchsize
                batch = data[low_:high_]
                x =  batch[:,:frameskip]
                act =  batch[:, frameskip].astype(int)
                r = batch[:, (frameskip+1)]

                inp = env.pos_to_image(x)
                out = -r

                yield [inp, np.eye(env.n_actions)[act]], out


    @threadsafe_generator
    def full_gen(self, env, all_idxs, batch_size=32):

        data_length = len(all_idxs)
        steps = int(np.ceil(data_length/float(batch_size)))

        states = self.data.states()
        states_ = self.data.next_states()
        lengths = self.data.lengths()
        rewards = self.data.rewards().reshape(-1)
        actions = self.data.actions().reshape(-1)
        dones = self.data.dones().reshape(-1)

        shp = states.shape
        states = states.reshape(np.prod(shp[:2]), -1)
        states_ = states_.reshape(np.prod(shp[:2]), -1)

        while True:
            perm = np.random.permutation(all_idxs)
            for batch in np.arange(steps):
                batch_idxs = perm[(batch*batch_size):((batch+1)*batch_size)]

                x = states[batch_idxs]
                x_ = states_[batch_idxs]
                r = rewards[batch_idxs]
                done = dones[batch_idxs]
                act = actions[batch_idxs]

                if self.modeltype in ['conv']:
                    tmp_shp = np.hstack([len(batch_idxs),-1,shp[2:]])
                    inp = self.processor(x.reshape(tmp_shp).squeeze())
                    out_x_ = np.diff(self.processor(x_.reshape(tmp_shp)).squeeze(), axis=1)[:,[-1],...]
                    out_r = -r
                    out_done = done
                elif self.modeltype == 'conv1':
                    tmp_shp = np.hstack([len(batch_idxs),-1,shp[2:]])
                    inp = self.processor(x.reshape(tmp_shp).squeeze())
                    inp = inp[:,None,:,:]
                    out_x_ = np.squeeze((x_-x).reshape(tmp_shp))
                    out_x_ = out_x_[:,None,:,:]
                    out_r = -r
                    out_done = done
                else:
                    tmp_shp = np.hstack([len(batch_idxs),-1,shp[2:]])
                    inp = np.squeeze(x.reshape(tmp_shp))
                    out_x_ = x_
                    out_x_ = np.diff(out_x_.reshape(tmp_shp), axis=2).reshape(-np.prod(tmp_shp[:2]), -1)
                    out_r = -r
                    out_done = done
                    out_x_ = out_x_[:,None,...]

                yield [inp, np.eye(env.n_actions)[act]], [out_x_, out_r, out_done]
                # yield ([x, np.eye(3)[acts], np.array(weight).reshape(-1,1)], [np.array(R).reshape(-1,1)])

    @staticmethod
    def compare(num_batches_val,val_gen,model):
        g  = []
        for j in range(num_batches_val):
            x, y = next(val_gen)
            arr = []
            for i in range(len(y)):
                arr.append(np.mean(np.mean((model.predict(x)[i] - y[i])**2, axis=-1)))
            g.append(arr)
        return g

    def run(self, env, dataset, num_epochs=100, batchsize=32, modeltype='conv'):
        '''
        Fit NN to transitions, rewards, and done
        '''

        # Fit P(s' | s,a)
        # Do this by change in pixels.
        self.modeltype = modeltype
        if self.modeltype in ['conv', 'mlp', 'conv1']:
            im = dataset.states()[0]
            if self.processor: im = self.processor(im)
            input_shape = im.shape[1:]#(self.frameheight, 2)

            full, full_all = self.create_full_model(input_shape)

            earlyStopping = EarlyStopping(monitor='val_loss', min_delta=1e-4,  patience=5, verbose=1, mode='min', restore_best_weights=True)
            reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_delta=1e-4, mode='min')


            self.data = dataset
            dataset_length = self.data.num_tuples()
            perm = np.random.permutation(range(dataset_length))
            eighty_percent_of_set = int(.8*len(perm))
            training_idxs = perm[:eighty_percent_of_set]
            validation_idxs = perm[eighty_percent_of_set:]
            training_steps_per_epoch = int(1.*np.ceil(len(training_idxs)/float(batchsize)))
            validation_steps_per_epoch = int(np.ceil(len(validation_idxs)/float(batchsize)))

            train_gen = self.full_gen(env, training_idxs, batchsize)
            val_gen = self.full_gen(env, validation_idxs, batchsize)

            hist = full.fit_generator(train_gen,
                                    steps_per_epoch=training_steps_per_epoch,
                                    validation_data=val_gen,
                                    validation_steps=validation_steps_per_epoch,
                                    epochs=num_epochs,
                                    callbacks=[earlyStopping, reduce_lr_loss],
                                    max_queue_size=1,
                                    workers=1,
                                    use_multiprocessing=False, )


            # try:
            #     print('Loading Full Model')
            #     full.load_weights(os.path.join(os.getcwd(), self.filename,'full.h5'))
            # except:
            #     print('Failed to load. Relearning')
            #     earlyStopping = EarlyStopping(monitor='val_loss', min_delta=1e-4,  patience=7, verbose=1, mode='min', restore_best_weights=True)
            #     # mcp_save = ModelCheckpoint('T.hdf5', save_best_only=True, monitor='val_loss', mode='min')
            #     reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1, min_delta=1e-5, mode='min')

            #     hist = full.fit_generator(train_gen,
            #                             steps_per_epoch=num_batches_train,
            #                             validation_data=val_gen,
            #                             validation_steps=num_batches_val,
            #                             epochs=num_epochs,
            #                             callbacks=[earlyStopping, reduce_lr_loss],
            #                             max_queue_size=1,
            #                             workers=4,
            #                             use_multiprocessing=False, )
            #     full.save_weights(os.path.join(os.getcwd(), self.filename,'full.h5'))
        else: # Linear


            x = dataset.states()
            act = dataset.actions().reshape(-1)
            r = dataset.rewards().reshape(-1)
            x_ = dataset.next_states()
            done = dataset.dones().reshape(-1)

            inp = x.reshape(np.prod(x.shape[:2]), -1)
            out_x_ = np.diff(x_, axis=2)

            out_x_ = out_x_.reshape(np.prod(out_x_.shape[:2]), -1)
            out_r = -r.reshape(-1)
            out_done = done.reshape(-1)

            X = np.hstack([inp, np.eye(3)[act]])

            full_T = LinearRegression().fit(X, out_x_)
            full_R = LinearRegression().fit(X, out_r)
            full_D = LogisticRegression().fit(X, out_done)

            full = [full_T, full_R, full_D]


        self.full = full
        return self


    def estimate_R(self, x, a, t):
        #Approximated rewards
        reward = -self.R.predict([x,a]).reshape(-1)

        return reward

    def estimate_R_all(self, x):
        return -self.R_all.predict([x])

    def old_transition(self, x, a):
        # Exact MDP dynamics
        # self.P = {(0, 0): {0: 0.5, 1: 0.5}, (0, 1): {0: 0.5, 1: .5}}

        #Approximated dynamics
        # if tuple([x,a]) in self.P:
        #     try:
        #         state = np.random.choice(list(self.P[(x,a)]), p=list(self.P[(x,a)].values()))
        #         if self.override_done:
        #             done = False
        #         else:
        #             done = np.random.choice(list(self.D[(x,a,state)]),
        #                                     p=list(self.D[(x,a,state)].values()))
        #     except:
        #         import pdb; pdb.set_trace()
        # else:
        #     state = None
        #     done = True
        state_diff = self.T.predict([x, a])
        x_ = np.concatenate([x[:,1:2,...], x[:,1:2,...] + state_diff], axis=1)

        prob_done = self.D.predict([np.concatenate([x, x_], axis=1), a])
        done = np.array([np.random.choice([0,1], p=[1-d[0], d[0]]) for d in prob_done])

        return x_, done


    def transition(self, x, a):
        if isinstance(self.full, list):
            state_diff, r, prob_done = [model.predict(np.hstack([x.reshape(x.shape[0],-1), a])) for model in self.full]
            state_diff = state_diff[:,None,:]
            prob_done = [[d] for d in prob_done]
        else:
            [state_diff, r, prob_done] = self.full.predict([x, a], batch_size=128)

        x_ = np.concatenate([x[:,1:self.frameheight,...], x[:,(self.frameheight-1):self.frameheight,...] + state_diff], axis=1)
        done = np.array([np.random.choice([0,1], p=[1-d[0], d[0]]) for d in prob_done])

        return x_, -r.reshape(-1), done

    def Q(self, policy, x, t=0):

        Qs = []

        # state = x
        # make action agnostic.
        state = np.repeat(x, self.action_space_dim, axis=0)
        acts  = np.tile(np.arange(self.action_space_dim), len(x))

        done = np.zeros(len(state))
        costs = []
        trajectory_length = t
        # Q
        cost_to_go = np.zeros(len(state))

        new_state, cost_holder, new_done = self.transition(state, np.atleast_2d(np.eye(self.action_space_dim)[acts]))
        # cost_holder = self.estimate_R(state, np.atleast_2d(np.eye(self.action_space_dim)[acts]), None)

        done = done + new_done
        new_cost_to_go = cost_to_go + self.gamma * cost_holder * (1-done)

        norm_change = np.sqrt(np.sum((new_cost_to_go-cost_to_go)**2) / len(state))
        # print(trajectory_length, norm_change, cost_to_go, sum(done), len(done))
        cost_to_go = new_cost_to_go

        if norm_change < 1e-4:
            done = np.array([True])

        trajectory_length += 1
        if self.max_traj_length is not None:
            if trajectory_length >= self.max_traj_length:
                done = np.array([True])

        state = new_state

        while not done.all():
            tic=time.time()

            still_alive = np.where(1-done)[0]
            acts = policy.sample(state[still_alive])

            new_state, cost_holder, new_done = self.transition(state[still_alive], np.atleast_2d(np.eye(self.action_space_dim)[acts]))

            # cost_holder = self.estimate_R(state, np.atleast_2d(np.eye(self.action_space_dim)[acts]), trajectory_length)
            # if (tuple([state,a,new_state]) in self.terminal_transitions):
            #     done = True

            done[still_alive] = (done[still_alive] + new_done).astype(bool)
            new_cost_to_go = cost_to_go[still_alive] + self.gamma * cost_holder * (1-done[still_alive])

            # norm_change = np.sqrt(np.sum((new_cost_to_go-cost_to_go)**2) / len(state))
            # print(trajectory_length, norm_change, cost_to_go, sum(done), len(done))
            cost_to_go[still_alive] = new_cost_to_go

            # if norm_change < 1e-4:
            #     done = np.array([True])

            trajectory_length += 1
            if self.max_traj_length is not None:
                if trajectory_length >= self.max_traj_length:
                    done = np.array([True])

            # print(time.time()-tic, trajectory_length)
            state[still_alive] = new_state

        return cost_to_go


    @staticmethod
    def discounted_sum(costs, discount):
        '''
        Calculate discounted sum of costs
        '''
        y = signal.lfilter([1], [1, -discount], x=costs[::-1])
        return y[::-1][0]
