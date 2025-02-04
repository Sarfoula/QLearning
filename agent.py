import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DQNAgent:
	def __init__(self):
		self.input_dim = 5
		self.output_dim = 3

		self.model = self.build_model()
		self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
		self.criterion = nn.MSELoss()

		self.discount_factor = 0.95
		self.exploration_prob = 0.2

	def build_model(self):
		model = nn.Sequential(
			nn.Linear(self.input_dim, 64),
			nn.ReLU(),
			nn.Linear(64, 64),
			nn.ReLU(),
			nn.Linear(64, self.output_dim)
		)
		return model

	def choose_action(self, state):
		state_tensor = torch.FloatTensor(state)
		if np.random.rand() < self.exploration_prob:
			return np.random.randint(0, self.output_dim)
		else:
			with torch.no_grad():
				q_values = self.model(state_tensor)
			return torch.argmax(q_values).item()

	def update(self, state, action, reward, next_state, done):
		state_tensor = torch.FloatTensor(state)
		next_state_tensor = torch.FloatTensor(next_state)

		q_values = self.model(state_tensor)
		q_value_current = q_values[action]

		if done:
			q_value_next = 0
		else:
			q_values_next = self.model(next_state_tensor)
			q_value_next = torch.max(q_values_next).item()

		target = reward + self.discount_factor * q_value_next

		loss = self.criterion(q_value_current, torch.FloatTensor([target]))

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

	def decrease_exploration_prob(self):
		if self.exploration_prob > 0.01:
			self.exploration_prob *= 0.995
