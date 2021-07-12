import numpy as np

class BasicPolicy(object):
	def __init__(self, actions, probs):
		self.actions = actions
		self.probs = np.array(probs)
		self.action_space_dim = len(self.actions)
		assert len(self.actions) == len(self.probs)

	def predict(self, xs, **kw):
		return np.array([self.probs for _ in range(len(xs))])

	def sample(self, xs):
		return self(xs)

	def __call__(self, states):
		return np.random.choice(self.actions, size=len(states), p=self.probs)


class BasicPolicy_MODELWIN(object):
	def __init__(self, actions, probs):
		self.actions = actions
		self.probs = np.array(probs)
		assert len(self.actions) == len(self.probs)

	def predict(self, xs, **kw):
		out = []
		for x in xs:
			if x != 0:
				out.append(np.ones(len(self.actions))/ len(self.actions))
			else:
				out.append(self.probs)

		return np.array(out)

	def sample(self, xs):
		return self(xs)

	def __call__(self, states):
		out = []
		for x in states:
			if x != 0:
				out.append(np.random.choice(self.actions))
			else:
				out.append(np.random.choice(self.actions, p=self.probs))

		return np.array(out)

class BasicQ(object):
    def __init__(self):
        pass

    def __call__(self, *args, **kw):
        return 1

class SingleTrajectory(object):
	def __init__(self, actions, trajectory):
		self.actions = actions
		self.trajectory = trajectory

	def predict(self, xs, **kw):
		out = []
		for x in xs:
			try:
				probs = np.zeros(len(self.actions))
				probs[self.trajectory[x]] = 1.
				out.append(probs)
			except:
				out.append([1.] + [0.]*(len(self.actions)-1))

		return np.array(out)


	def __call__(self, states):
		out = []
		for x in states:
			try:
				out.append(self.trajectory[x])
			except:
				out.append(0)

		return np.array(out)
