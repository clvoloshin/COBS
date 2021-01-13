import numpy as np
import scipy.signal as signal

class ModelFail(object):
	def __init__(self,
				 make_pomdp = False,
				 number_of_pomdp_states = 2,
				 transitions_deterministic=True,
				 max_length = 2,
				 sparse_rewards = False,
				 stochastic_rewards = False):
		self.allowable_actions = [0,1]
		self.n_actions = len(self.allowable_actions)
		self.n_dim = 2*max_length



		self.make_pomdp = make_pomdp
		self.number_of_pomdp_states = number_of_pomdp_states

		split = np.array_split(np.arange(2, 2*max_length)-1, number_of_pomdp_states-1)

		self.state_to_pomdp_state = {}
		for pomdp_state,states in enumerate(split):
			for state in states:
				self.state_to_pomdp_state[state] = pomdp_state

		self.state_to_pomdp_state[0] = 0
		self.state_to_pomdp_state[2*max_length-1] = number_of_pomdp_states-1

		print(self.state_to_pomdp_state)
		self.transitions_deterministic = transitions_deterministic
		self.slippage = .25
		self.max_length = max_length
		self.sparse_rewards = sparse_rewards
		self.stochastic_rewards = stochastic_rewards
		self.reset()

	def num_states(self):
		return self.n_dim

	def pos_to_image(self, x):
		'''latent state -> representation '''
		return x

	def reset(self):
		self.state = 0
		self.done = False
		return np.array([self.state])

	def step(self, action):
		assert action in self.allowable_actions
		assert not self.done, 'Episode Over'
		reward = 0 if not self.stochastic_rewards else np.random.randn()


		if self.state == (2*self.max_length-3):
			reward = 1 if not self.stochastic_rewards else np.random.randn()+1
			# reward = 0
			self.state = 0 #2*self.max_length-1
			self.done = True
		elif self.state == (2*self.max_length-2):
			reward = -1 if not self.stochastic_rewards else np.random.randn()-1
			# reward = 0
			self.state = 0 #2*self.max_length-1
			self.done = True
		else:
			if self.state == 0:
				if action == 0:
					if self.transitions_deterministic:
						self.state = self.state + 1
					else:
						self.state = int(np.random.choice([self.state+1,self.state+2], p = [1-self.slippage,self.slippage]))
				else:
					if self.transitions_deterministic:
						self.state = self.state + 2
					else:
						self.state = int(np.random.choice([self.state+2,self.state+1], p = [1-self.slippage,self.slippage]))
			else:
				if action == 0:
					if self.transitions_deterministic:
						if self.state % 2 == 1:
							self.state = self.state + 2
						else:
							self.state = self.state + 1
					else:
						if self.state % 2 == 1:
							self.state = int(np.random.choice([self.state+2,self.state+3], p = [1-self.slippage,self.slippage]))
						else:
							self.state = int(np.random.choice([self.state+1,self.state+2], p = [1-self.slippage,self.slippage]))
				else:
					if self.transitions_deterministic:
						if self.state % 2 == 1:
							self.state = self.state + 3
						else:
							self.state = self.state + 2
					else:
						if self.state % 2 == 1:
							self.state = int(np.random.choice([self.state+3,self.state+2], p = [1-self.slippage,self.slippage]))
						else:
							self.state = int(np.random.choice([self.state+2,self.state+1], p = [1-self.slippage,self.slippage]))

			if not self.sparse_rewards:


				if self.state % 2 == 1:
					rew = 1 if not self.stochastic_rewards else np.random.randn()+1
					reward = rew
				else:
					rew = -1 if not self.stochastic_rewards else np.random.randn()-1
					reward = rew
			# else:
			# 	if self.state == 2*self.max_length-3:
			# 		reward = 1
			# 	elif self.state == 2*self.max_length-2:
			# 		reward = -1
			# 	else:
			# 		reward = 0

				# reward = 1 if self.state == 2*self.max_length-1

		state = self.state_to_pomdp_state[self.state]

		if self.make_pomdp:
			# only reveal state, not internal state (POMDP)
			return np.array([state]), reward, self.done, {}
		else:
			return np.array([self.state]), reward, self.done, {}

	def render(self, a=None, r=None, return_arr=False):
		start_state = 1 if self.state == 0 else 0
		state = np.zeros(2*self.max_length-2)
		end_state = 1 if self.state == (2*self.max_length-1) else 0

		if not start_state and not end_state:
			state[self.state-1] = 1

		if return_arr:
			return start_state, state.reshape(2,self.max_length-1, order='F'), end_state
		else:

			print(' ', ' '.join(state.reshape(2,self.max_length-1, order='F')[0].astype(int).astype(str).tolist()), '  ')
			if (a is not None) and (r is not None):
				print(start_state, ' '*((2*(self.max_length-2))+1), end_state, ' (a,r): ', (a,r), '.  If POMDP, End state: ', end_state)
			else:
				print(start_state, ' '*((2*(self.max_length-2))+1), end_state)
			print(' ', ' '.join(state.reshape(2,self.max_length-1, order='F')[1].astype(int).astype(str).tolist()), '  ')
			print('\n')
			# print([start_state], [end_state], state.reshape(2,self.max_length-1, order='F'), )

	def calculate_exact_value_of_policy(self, pi_e, gamma):
		# Exact
		# rewards = []
		# if (self.transitions_deterministic):
		# 	rew = [(+1)*(pi_e.probs[0]) + (-1)*(pi_e.probs[1])]
		# 	if not self.sparse_rewards:
		# 		rewards.append(rew*self.max_length)
		# 	else:
		# 		rewards.append([0]*(self.max_length-1) + rew)

		# else:
		# 	rew = [(+1)*(pi_e.probs[0]*(1-self.slippage) + pi_e.probs[1]*(self.slippage)) + (-1)*(pi_e.probs[0]*(self.slippage) + pi_e.probs[1]*(1-self.slippage))]
		# 	if not self.sparse_rewards:
		# 		rewards.append(rew*self.max_length)
		# 	else:
		# 		rewards.append([0]*(self.max_length-1) + rew)

		# Approx
		evaluation = []
		for i in range(5000):
		    done = False
		    state = self.reset()
		    # env.render()
		    rewards = []
		    while not done:
		        action = pi_e([state])
		        # print(action)
		        next_state, reward, done = self.step(action)
		        # env.render()
		        state = next_state
		        rewards.append(reward)

		    evaluation.append(rewards)

		true = np.mean([self.discounted_sum(rew, gamma) for rew in np.array(evaluation)])

		return true

	@staticmethod
	def discounted_sum(costs, discount):
	    '''
	    Calculate discounted sum of costs
	    '''
	    y = signal.lfilter([1], [1, -discount], x=costs[::-1])
	    return y[::-1][0]



