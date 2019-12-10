import numpy as np

class ModelWin(object):
	def __init__(self, 
				 make_pomdp = False, 
				 number_of_pomdp_states = 2,
				 transitions_deterministic=True,
				 max_length = 2,
				 sparse_rewards = False):
		self.make_pomdp = make_pomdp
		self.number_of_pomdp_states = number_of_pomdp_states
		self.transitions_deterministic = transitions_deterministic
		self.max_length = max_length
		self.sparse_rewards = sparse_rewards
		self.reset()


	def pos_to_image(self, x):
		'''latent state -> representation '''
		return x
		
	def reset(self):
		self.state = 0
		self.done = False
		self.t = 0
		return self.state

	def step(self, action):
		assert action in [0,1]
		assert not self.done, 'Episode Over'
		reward = 0


		if self.state == 0:
			if action == 0:
				new_state = np.random.choice([1,2], p=[.4, .6])
				if new_state == 1:
					reward = 1
				else:
					reward = -1
			else:
				new_state = np.random.choice([1,2], p=[.6, .4])
				if new_state == 1:
					reward = 1
				else:
					reward = -1
		if self.state != 0:
			new_state = 0 

		self.state = new_state

		self.t += 1
		if self.t >= self.max_length:
			# self.state = 3
			self.done = True

		return self.state, reward, self.done

	def render(self, *args, return_arr=False):
		print([self.state == 1, self.state == 0, self.state == 2, self.state == 3])

