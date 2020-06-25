import numpy as np

class Hybrid(object):
	def __init__(self, 
				 make_pomdp = False, 
				 transitions_deterministic=True,
				 max_length = 22,
				 sparse_rewards = False):
		self.make_pomdp = make_pomdp
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


		if self.state == 3:
			if action == 0:
				new_state = np.random.choice([4,5], p=[.4, .6])
			else:
				new_state = np.random.choice([4,5], p=[.6, .4])
            reward = 1 if new_state == 4 else -1

        elif self.state == 0:
            new_state = 1 if action == 0 else 2

		elif self.state in [1,2]:
            reward = 1 if self.state == 1 else -1
			new_state = 3

        else:
            new_state = 3

		self.state = new_state

		self.t += 1
		if self.t >= self.max_length:
			self.done = True

		return self.state, reward, self.done

	def render(self, *args, return_arr=False):
		print("Not Implemented")