
from tqdm import trange
import numpy as np

class getQs(object):
	def __init__(self, data, pi_e, processor, action_space_dim = 3):
		self.data = data
		self.pi_e = pi_e
		self.processor = processor
		self.action_space_dim = action_space_dim

	def get(self, model):
		Qs = []
		batchsize = 1
		num_batches = int(np.ceil(len(self.data)/batchsize))
		# frames = np.array([x['frames'] for x in self.trajectories])
		for batchnum in trange(num_batches, desc='Batch'):
			low_ = batchsize*batchnum
			high_ = min(batchsize*(batchnum+1), len(self.data))

			pos = self.data.states(False, low_=low_,high_=high_)
			acts = self.data.actions()[low_:high_]


			# episodes = self.trajectories[low_:high_]
			# pos = np.vstack([np.vstack(x['x']) for x in episodes])
			# N = np.hstack([[low_ + n]*len(x['x']) for n,x in enumerate(episodes)])
			# acts = np.hstack([x['a'] for x in episodes])
			# pos = np.array([np.array(frames[int(N[idx])])[pos[idx].astype(int)] for idx in range(len(pos))])
			traj_Qs = model.Q(self.pi_e, self.processor(pos))

			traj_Qs = traj_Qs.reshape(-1, self.action_space_dim)
			# lengths = self.data.lengths()

			# endpts = np.cumsum(np.hstack([[0], lengths]))
			# for start,end in zip(endpts[:-1], endpts[1:]):
			# 	Qs.append(traj_Qs[start:end])
			Qs.append(traj_Qs)

		return Qs
