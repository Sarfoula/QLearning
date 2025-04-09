import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils as utils
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
			priority = (abs(priority) + self.epsilon) ** self.alpha
			self.tree.update(idx, priority)
			self.max_priority = max(self.max_priority, priority)

class DuelingDeepQNetwork(nn.Module):
	def __init__(self, input_dim, output_dim, lr, fc1_dims=64, fc2_dims=64):
		super().__init__()
		self.fc1 = nn.Linear(input_dim, fc1_dims)
		self.fc2 = nn.Linear(fc1_dims, fc2_dims)

		self.value_stream = nn.Linear(fc2_dims, 1)

		self.advantage_stream = nn.Linear(fc2_dims, output_dim)

		self.loss = nn.SmoothL1Loss(reduction='none')
		self.optimizer = optim.Adam(self.parameters(), lr=lr)

	def forward(self, state):
		x = F.relu(self.fc1(state))
		x = F.relu(self.fc2(x))

		value = self.value_stream(x)
		advantages = self.advantage_stream(x)

		q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
		return q_values

class Agent():
	def __init__(self, input_dim, output_dim, batch_size=64, capacity=500000):
		self.gamma = 0.99
		self.lr = 0.0003
		self.tau = 0.005
		self.step_counter = 0

		self.epsilon = 1.0
		self.epsilon_min = 0.1
		self.epsilon_decay_episode = 300
		self.decay_rate = (self.epsilon - self.epsilon_min) / self.epsilon_decay_episode

		self.replay_buffer = ReplayBuffer(capacity, batch_size, alpha=0.6, beta=0.4, beta_inc=0.001)
		self.policy = DuelingDeepQNetwork(input_dim, output_dim, self.lr, fc1_dims=64, fc2_dims=64)
		self.target = DuelingDeepQNetwork(input_dim, output_dim, self.lr, fc1_dims=64, fc2_dims=64)
		self.target.load_state_dict(self.policy.state_dict())
		self.target.eval()

	def best_action(self, state):
		with T.no_grad():
			tensor = T.tensor(state, dtype=T.float32).unsqueeze(0)
			action = T.argmax(self.policy.forward(tensor)).item()
		return action

	def epsilon_decay(self):
		self.epsilon = max(self.epsilon_min, self.epsilon - self.decay_rate)

	def choose_action(self, state):
		if random.random() < self.epsilon:
			action = random.randint(0, 2)
		else:
			with T.no_grad():
				tensor = T.tensor(state, dtype=T.float32).unsqueeze(0)
				action = T.argmax(self.policy.forward(tensor)).item()
		return action

	def learn(self):
		if self.replay_buffer.tree.size < self.replay_buffer.batch_size:
			return

		states, actions, rewards, new_states, terminals, batch_indices, weights = self.replay_buffer.sample()

		q_values = self.policy(states)
		q_value = q_values.gather(1, actions).squeeze(1)

		with T.no_grad():
			next_max_action = self.policy(new_states).argmax(1, keepdim=True)
			max_next_q_value = self.target(new_states).gather(1, next_max_action).squeeze(1)
			q_target = rewards + self.gamma * max_next_q_value * (1 - terminals)

		loss = (self.policy.loss(q_value, q_target) * weights).mean()

		td_errors = T.abs(q_value - q_target).detach().numpy()
		self.replay_buffer.update_priorities(batch_indices, td_errors)

		self.policy.optimizer.zero_grad()
		loss.backward()
		utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
		self.policy.optimizer.step()

		self.step_counter += 1
		for target_param, policy_param in zip(self.target.parameters(), self.policy.parameters()):
			target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)
		return loss.item()

	def save_model(self, path):
		T.save({'model_state_dict': self.policy.state_dict(),
			'target_state_dict': self.target.state_dict(),
			'optimizer_state_dict': self.policy.optimizer.state_dict(),
			'epsilon': self.epsilon}, path)

	def load_model(self, path):
		checkpoint = T.load(path, weights_only=False)
		self.policy.load_state_dict(checkpoint['model_state_dict'])
		self.target.load_state_dict(checkpoint['target_state_dict'])
		self.policy.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		self.epsilon = checkpoint['epsilon']
