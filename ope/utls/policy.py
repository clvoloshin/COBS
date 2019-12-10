import numpy as np
import gym
from scipy.stats import truncnorm

# Set env and random seed
#np.random.seed(1)
#env = gym.make('Pendulum-v0')
#env = env.unwrapped

# Global Dimension
obs_dim = 3 #env.observation_space.shape[0]
act_dim = 1 #env.action_space.shape[0]
hidden_dim = 5

class Gaussian_policy(object):
	def Relu(self, x):
		return np.maximum(x,0)
	def theta_to_gp(self,theta):
		t = 0
		W1 = np.array(theta[t:t+obs_dim*hidden_dim], dtype = np.float32)
		W1 = W1.reshape((obs_dim, hidden_dim))
		t = t+obs_dim*hidden_dim
		b1 = np.array(theta[t:t+hidden_dim], dtype = np.float32)
		t = t+hidden_dim
		W2 = np.array(theta[t:t+hidden_dim*act_dim], dtype = np.float32)
		W2 = W2.reshape((hidden_dim, act_dim))
		t = t+hidden_dim*act_dim
		b2 = np.array(theta[t:t+act_dim], dtype = np.float32)
		t = t+act_dim
		logvar = theta[t]
		return W1,b1,W2,b2,logvar
	def __init__(self, theta):
		self.W1, self.b1, self.W2, self.b2, self.logvar = self.theta_to_gp(theta)

	def get_mean(self, state):
		return np.matmul(self.Relu(np.matmul(state, self.W1) + self.b1), self.W2) + self.b2

	def choose_action(self, single_state):
		state = single_state[None,:]
		mean = self.get_mean(state)
		mean = mean.reshape([-1])
		return mean + np.exp(self.logvar/2) * np.random.randn(act_dim)

	def log_pi(self, states, actions):
		N = states.shape[0]
		states = states.reshape(-1,obs_dim)
		actions = actions.reshape(-1,act_dim)
		mean = self.get_mean(states)
		diff = mean - actions
		ret = -0.5*self.logvar - 0.5*np.sum(diff*diff,axis = -1)/np.exp(self.logvar)
		return ret.reshape(N,-1)

class Truncated_Gaussian_policy(object):
	def Relu(self, x):
		return np.maximum(x,0)
	def theta_to_gp(self,theta):
		t = 0
		W1 = np.array(theta[t:t+obs_dim*hidden_dim], dtype = np.float32)
		W1 = W1.reshape((obs_dim, hidden_dim))
		t = t+obs_dim*hidden_dim
		b1 = np.array(theta[t:t+hidden_dim], dtype = np.float32)
		t = t+hidden_dim
		W2 = np.array(theta[t:t+hidden_dim*act_dim], dtype = np.float32)
		W2 = W2.reshape((hidden_dim, act_dim))
		t = t+hidden_dim*act_dim
		b2 = np.array(theta[t:t+act_dim], dtype = np.float32)
		t = t+act_dim
		logvar = theta[t]
		return W1,b1,W2,b2,logvar
	def __init__(self, theta, epsilon):
		self.W1, self.b1, self.W2, self.b2, self.log_scale = self.theta_to_gp(theta)
		self.epsilon = epsilon
	def get_mean(self, state):
		return np.matmul(self.Relu(np.matmul(state, self.W1) + self.b1), self.W2) + self.b2

	def sample(self, single_state):
		return self.choose_action(single_state)

	def choose_action(self, single_state):
		if np.random.rand()<self.epsilon:
			return np.random.rand(1)*8 - 4.0
		state = single_state[None, :]
		loc = self.get_mean(state)
		loc = loc.item(0)
		scale = np.exp(self.log_scale)
		a = (-4.0-loc)/scale
		b = (4.0-loc)/scale
		return [truncnorm.rvs(a,b,loc = loc, scale = scale)]

	def log_pi(self, states, actions):
		N = states.shape[0]
		states = states.reshape(-1,obs_dim)
		actions = actions.reshape(-1,act_dim)
		locs = self.get_mean(states)
		scale = np.exp(self.log_scale)
		n = states.shape[0]
		pdf = np.zeros([n], dtype = np.float32)
		for i in range(n):
			a = (-4.0-locs[i])/scale
			b = (4.0-locs[i])/scale
			pdf[i] = truncnorm.pdf(actions[i], a, b, loc = locs[i], scale = scale)
		ret = np.log((1-self.epsilon)*pdf + self.epsilon/8.0)
		return ret.reshape(N,-1)

class Mixed_Policy(object):
	def __init__(self, policy0, policy1, ratio):
		self.p1 = policy1
		self.p0 = policy0
		self.ratio = ratio
	def choose_action(self, state):
		if np.random.rand() < self.ratio:
			return self.p0.choose_action(state)
		else:
			return self.p1.choose_action(state)
	def log_pi(self, states, actions):
		log_pi0 = self.p0.log_pi(states, actions)
		log_pi1 = self.p1.log_pi(states, actions)
		return np.log(self.ratio * np.exp(log_pi0) + (1-self.ratio) * np.exp(log_pi1))
