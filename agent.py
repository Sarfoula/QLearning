import torch as T
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class SumTree:
	def __init__(self, capacity):
		self.capacity = capacity
		self.tree = np.zeros(2 * capacity - 1)
		self.data = np.zeros(capacity, dtype=object)
		self.size = 0
		self.ptr = 0

	def add(self, priority, sample):
		idx = self.ptr + self.capacity - 1
		self.data[self.ptr] = sample
		self.update(idx, priority)

		self.ptr = (self.ptr + 1) % self.capacity
		self.size = min(self.size + 1, self.capacity)

	def update(self, idx, priority):
		change = priority - self.tree[idx]
		self.tree[idx] = priority
		while idx != 0:
			idx = (idx - 1) // 2
			self.tree[idx] += change

	def sample(self, s):
		idx = 0
		while idx < self.capacity - 1:
			left, right = 2 * idx + 1, 2 * idx + 2
			if s <= self.tree[left]:
				idx = left
			else:
				s -= self.tree[left]
				idx = right
		return idx, self.tree[idx], self.data[idx - (self.capacity - 1)]

	def total_priority(self):
		return self.tree[0]

class ReplayBuffer:
	def __init__(self, buffer_size, batch_size, alpha, beta, beta_inc):
		self.tree = SumTree(buffer_size)
		self.buffer_size = buffer_size
		self.batch_size = batch_size
		self.alpha = alpha
		self.beta = beta
		self.bet_incr = beta_inc
		self.max_priority = 1.0
		self.epsilon = 1e-5

	def store_transition(self, state, action, reward, next_state, done):
		priority = self.max_priority
		self.tree.add(priority, (state, action, reward, next_state, done))

	def sample(self):
		batch_indices, batch_transitions, batch_weigths = [], [], []

		segment = self.tree.total_priority() / self.batch_size
		self.beta = min(1.0, self.beta + self.bet_incr)

		for i in range(self.batch_size):
			a, b = segment * i, segment * (i + 1)
			s = random.uniform(a, b)
			idx, p, data = self.tree.sample(s)

			prob = p / self.tree.total_priority()
			weight = (prob * self.tree.size) ** -self.beta

			batch_indices.append(idx)
			batch_transitions.append(data)
			batch_weigths.append(weight)

		batch_weigths = np.array(batch_weigths) / np.max(batch_weigths)

		states, actions, rewards, next_states, dones = zip(*batch_transitions)

		states = T.tensor(np.array(states), dtype=T.float32)
		next_states = T.tensor(np.array(next_states), dtype=T.float32)
		actions = T.tensor(actions, dtype=T.long).unsqueeze(1)
		rewards = T.tensor(rewards, dtype=T.float32)
		dones = T.tensor(dones, dtype=T.float32)
		weights = T.tensor(batch_weigths, dtype=T.float32)

		return states, actions, rewards, next_states, dones, batch_indices, weights

	def update_priorities(self, batch_indices, batch_priorities):
		for idx, priority in zip(batch_indices, batch_priorities):
			safe_priority = max(priority + self.epsilon, self.epsilon)
			self.tree.update(idx, safe_priority)
			self.max_priority = max(self.max_priority, safe_priority)

class DeepQNetwork(nn.Module):
	def __init__(self, input_dim, output_dim, lr, fc1_dims=64, fc2_dims=64):
		super().__init__()
		self.network = nn.Sequential(
		nn.Linear(input_dim, fc1_dims),
		nn.ReLU(),
		nn.Linear(fc1_dims, fc2_dims),
		nn.ReLU(),
		nn.Linear(fc2_dims, output_dim))

		self.loss = nn.SmoothL1Loss(reduction='none')
		self.optimizer = optim.Adam(self.parameters(), lr=lr)

	def forward(self, state):
		return self.network(state)

class Agent():
	def __init__(self, input_dim, output_dim, batch_size=64):
		# discount factor for the Bellman equation
		self.gamma = 0.99

		self.lr = 0.00025
		# Number of step for update target network
		self.tau = 1000
		self.step_counter = 0

		# epsilon greedy policy for exploration, decrease over time by deacay_rate and never goes below min
		self.epsilon = 1
		self.epsilon_min = 0.001
		self.decay_rate = 0.99997

		# Size for batch sample to learn from
		self.replay_buffer = ReplayBuffer(100000, batch_size, alpha=0.6, beta=0.4, beta_inc=0.001)

		# 2 neural network for the actual policy and for the target
		self.policy = DeepQNetwork(input_dim, output_dim, self.lr, fc1_dims=64, fc2_dims=64)
		self.target = DeepQNetwork(input_dim, output_dim, self.lr, fc1_dims=64, fc2_dims=64)
		self.target.load_state_dict(self.policy.state_dict())
		self.target.eval()

	def real_choose(self, state):
		return T.argmax(self.policy.forward(T.tensor(state, dtype=T.float32))).item()

	def epsilon_decay(self):
		"""Decrease the epsilon exponentioly by a certain decay rate"""
		self.epsilon = max(self.epsilon_min, self.epsilon * self.decay_rate)

	def choose_action(self, state):
		"""Choose action with epsilon greedy"""
		if random.random() < self.epsilon:
			action = random.randint(0, 2)
		else:
			with T.no_grad():
				action = T.argmax(self.policy.forward(T.FloatTensor(state))).item()
		self.epsilon_decay()
		return action

	def learn(self):
		"""Make the agent learn from a batch_size experience"""
		if self.replay_buffer.tree.size < self.replay_buffer.batch_size:
			return

		# Gives a random batch transition
		states, actions, rewards, new_states, terminals, batch_indices, weights = self.replay_buffer.sample()

		# Get q_value for state with the action choosen
		q_values = self.policy(states)
		q_value = q_values.gather(1, actions).squeeze(1)

		# Get best next_q_value for next_state with best next_action
		with T.no_grad():
			next_max_action = self.policy(new_states).argmax(1, keepdim=True)
			max_next_q_value = self.target(new_states).gather(1, next_max_action).squeeze(1)
			# Bellman equation gives use the target q_value (the ideal q_value)
			q_target = rewards + self.gamma * max_next_q_value * (1 - terminals)

		# The loss function is a difference between q_value and the q_target
		loss = (self.policy.loss(q_value, q_target) * weights).mean()

		td_errors = T.abs(q_value - q_target).detach().numpy()
		self.replay_buffer.update_priorities(batch_indices, td_errors)

		# Change the weight in the neural network to fit the q_target
		self.policy.optimizer.zero_grad()
		loss.backward()
		self.policy.optimizer.step()

		# Update the target network every tau step
		self.step_counter += 1
		if self.step_counter % self.tau == 0:
			self.target.load_state_dict(self.policy.state_dict())

	def save_model(self, path):
		T.save({'model_state_dict': self.policy.state_dict(),
		  'optimizer_state_dict': self.policy.optimizer.state_dict()}, path)

	def load_model(self, path):
		checkpoint = T.load(path)
		self.policy.load_state_dict(checkpoint['model_state_dict'])
		self.policy.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		self.policy.eval()
