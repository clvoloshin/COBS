from tqdm import trange
import numpy as np
import scipy.signal as signal

class QWrapper(object):
    def __init__(self, Qfunc):
        self.Q_values = Qfunc
    
    def get_Qs_for_data(self, data, cfg):
        Qs = []
        batchsize = 1
        num_batches = int(np.ceil(len(data)/batchsize))
        # frames = np.array([x['frames'] for x in self.trajectories])
        for batchnum in trange(num_batches, desc='Batch'):
            low_ = batchsize*batchnum
            high_ = min(batchsize*(batchnum+1), len(data))

            pos = data.states(False, low_=low_,high_=high_)
            acts = data.actions()[low_:high_]


            # episodes = self.trajectories[low_:high_]
            # pos = np.vstack([np.vstack(x['x']) for x in episodes])
            # N = np.hstack([[low_ + n]*len(x['x']) for n,x in enumerate(episodes)])
            # acts = np.hstack([x['a'] for x in episodes])
            # pos = np.array([np.array(frames[int(N[idx])])[pos[idx].astype(int)] for idx in range(len(pos))])
            traj_Qs = self.Q(data, cfg.processor(pos))

            traj_Qs = traj_Qs.reshape(-1, data.n_actions)
            # lengths = self.data.lengths()

            # endpts = np.cumsum(np.hstack([[0], lengths]))
            # for start,end in zip(endpts[:-1], endpts[1:]):
            # 	Qs.append(traj_Qs[start:end])
            Qs.append(traj_Qs)

        return Qs


    def Q(self, data, x, t=0):
        if self.Q_values.fitted == 'tabular':
            Qs = []
            for state in np.squeeze(x):
                # Qs.append(self.Q_values[self.map[state]])
                Qs.append(self.Q_values.predict(state))
            return  np.array(Qs)
        else:
            if self.Q_values.fitted != 'linear':
                return self.Q_values.predict(x)
            else:
                inp = np.repeat(x, data.n_actions, axis=0)
                act = np.tile(np.arange(data.n_actions), len(x))
                inp = np.hstack([inp.reshape(inp.shape[0],-1), np.eye(data.n_actions)[act]])
                val= self.Q_values.predict(inp).reshape(-1, data.n_actions)
                return val



    def V(self, policy, x, t=0):
        if not self.is_model:
            return np.sum([self.Q_values[self.map[x],act]*prob for act,prob in enumerate(policy.predict([x])[0])])
        else:
            if data.n_actions is None:
                data.n_actions = len(policy.predict(x)[0])
            return np.sum([self.Q_values.predict([x,np.eye(data.n_actions)[[act]]])[0][0]*prob for act,prob in enumerate(policy.predict(x)[0])])
