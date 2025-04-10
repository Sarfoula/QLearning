import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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

			if self.tree[left] <= 0:
				idx = right
			elif self.tree[right] <= 0:
				idx = left
			elif s <= self.tree[left]:
				idx = left
			else:
				s -= self.tree[left]
				idx = right
		return idx, self.tree[idx], self.data[idx - (self.capacity - 1)]

	def total_priority(self):
		return self.tree[0]

class ReplayBuffer:
	def __init__(self, capacity, batch_size, alpha=0.6, beta=0.4, beta_inc=0.001):
		self.tree = SumTree(capacity)
		self.batch_size = batch_size
		self.alpha = alpha
		self.beta = beta
		self.beta_inc = beta_inc
		self.max_priority = 1.0
		self.epsilon = 1e-5

	def store_transition(self, state, next_state):
		"""Stocke une transition (state, next_state) avec priorité maximale"""
		priority = self.max_priority
		self.tree.add(priority, (state, next_state))

	def sample(self):
		batch_indices, batch_states, batch_next_states, batch_weights = [], [], [], []

		total_priority = self.tree.total_priority()
		if total_priority == 0:
			return None

		segment = total_priority / self.batch_size
		self.beta = min(1.0, self.beta + self.beta_inc)

		for i in range(self.batch_size):
			a, b = segment * i, segment * (i + 1)
			s = random.uniform(a, b)

			idx, priority, data = self.tree.sample(s)

			prob = priority / total_priority
			weight = (prob * self.tree.size) ** -self.beta

			batch_indices.append(idx)
			state, next_state = data
			batch_states.append(state)
			batch_next_states.append(next_state)
			batch_weights.append(weight)

		batch_weights = np.array(batch_weights) / np.max(batch_weights)

		states = T.stack(batch_states)
		next_states = T.stack(batch_next_states)
		weights = T.tensor(batch_weights, dtype=T.float32)

		return states, next_states, batch_indices, weights

	def update_priorities(self, batch_indices, batch_priorities):
		"""Met à jour les priorités des transitions échantillonnées"""
		for idx, priority in zip(batch_indices, batch_priorities):
			priority = (abs(priority) + self.epsilon) ** self.alpha
			self.tree.update(idx, priority)
			self.max_priority = max(self.max_priority, priority)
class Network(nn.Module):
	def __init__(self, input_dim, output_dim, lr, fc1_dims=64, fc2_dims=64):
		super().__init__()
		self.model = nn.Sequential(nn.Linear(input_dim, fc1_dims),
							 nn.ReLU(),
							 nn.Linear(fc1_dims, fc2_dims),
							 nn.ReLU(),
							 nn.Linear(fc2_dims, fc1_dims),
							 nn.ReLU(),
							 nn.Linear(fc2_dims, output_dim))

	def forward(self, input):
		return self.model(input)

class Agent:
	def __init__(self, input_dim, output_dim, batch_size=64, hidden_dim=64, capacity=500000):
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.batch_size = batch_size

		self.agent = Network(input_dim, output_dim, lr=0.001, fc1_dims=hidden_dim, fc2_dims=hidden_dim)
		self.optim = optim.Adam(self.agent.parameters(), lr=0.001)
		self.loss = nn.MSELoss(reduction='none')

		self.buffer = ReplayBuffer(capacity, batch_size, alpha=0.6, beta=0.4, beta_inc=0.001)

	def save_model(self, path="save.pt"):
		T.save({'model_state_dict': self.agent.state_dict(),
		  'optimizer_state_dict': self.optim.state_dict()}, path)

	def load_model(self, path="save.pt"):
		file = T.load(path)
		self.agent.load_state_dict(file['model_state_dict'])
		self.optim.load_state_dict(file['optimizer_state_dict'])
		self.agent.eval()

	def store_transition(self, state, next_state):
		self.buffer.store_transition(state, next_state)

	def extract_position(self, state):
		pos_xy = state[1:3]
		return pos_xy

	def get_actions(self, state):
		paddle_y = state[0]
		with T.no_grad():
			prediction = self.agent(state)
		distance = (paddle_y - prediction[1]).item()

		if distance < 0:
			action = 1
		elif distance > 0:
			action = 0
		else:
			action = 2

		actions = [2] * 60
		max_move = min(int(abs(distance)//10), 60)

		for i in range(max_move):
			actions[i] = action
		return actions

	def learn(self):
		if self.buffer.tree.size < self.batch_size:
			return None

		batch = self.buffer.sample()
		if batch is None:
			return None

		states, next_states, indices, weights = batch

		next_positions = []
		for state in next_states:
			next_positions.append(self.extract_position(state))

		target_positions = T.stack(next_positions)
		predictions = self.agent(states)
		errors = self.loss(predictions, target_positions)
		weighted_loss = T.mean(errors * weights.unsqueeze(1))

		self.optim.zero_grad()
		weighted_loss.backward()
		self.optim.step()

		with T.no_grad():
			priorities = errors.sum(dim=1).detach().cpu().numpy()
			self.buffer.update_priorities(indices, priorities)

		return weighted_loss.item()
