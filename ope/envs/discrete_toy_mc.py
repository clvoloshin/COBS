import numpy as np

class DiscreteToyMC(object):
	def __init__(self, n_left = 10, n_right = 10, random_start = False):
		self.n_left = n_left
		self.n_right = n_right

		self.n_total_state = n_left + n_right + 2
		self.random_start = random_start

		self.n_actions = 2
		self.n_dim = self.n_total_state

		self.reset()

	def num_states(self):
		return self.n_total_state

	def step(self, action):
		assert not self.done

		if action == 0:
			if self.state != -self.n_left:
				self.state -= 1
		else:
			self.state += 1

		if self.state == (self.n_right + 1):
			self.done = True
			# self.state = 0

		# reward = -1
		# reward = 0 if action == 0 else 1
		# reward = 0 if not self.done else 1
		reward = -1 #if not self.done else 0

		return np.array([self.state+self.n_left]), reward, self.done, {}

	def reset(self):
		if self.random_start:
			self.state = np.random.choice(self.n_total_state-1) - self.n_left
		else:
			self.state = 0
		self.done = False
		return np.array([self.state+self.n_left])

	def render(self):
		out = ['_']*self.n_total_state
		out[self.state+self.n_left] = 'x'
		print(' '.join(out))

	def get_num_states(self):
		return self.n_total_state
