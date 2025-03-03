import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from replaybuffer import *

class DeepQNetwork(nn.Module):
	def __init__(self, lr, input_dims, n_actions, fc1_dims=64, fc2_dims=64):
		super(DeepQNetwork, self).__init__()
		self.fc1 = nn.Linear(input_dims, fc1_dims)
		self.fc2 = nn.Linear(fc1_dims, fc2_dims)
		self.fc3 = nn.Linear(fc2_dims, n_actions)

		self.optimizer = optim.Adam(self.parameters(), lr=lr)
		self.loss = nn.SmoothL1Loss()

	def forward(self, state):
		x = F.relu(self.fc1(state))
		x = F.relu(self.fc2(x))
		return self.fc3(x)

class dqn_rnn(nn.Module):
	def __init__(self, lr, input_dims, n_actions, fc1_dims=128, fc2_dims=512):
		super(dqn_rnn, self).__init__()
		self.fc1 = nn.LSTM(input_dims, fc1_dims, batch_first=True)
		self.fc2 = nn.Linear(fc1_dims, fc2_dims)
		self.fc3 = nn.Linear(fc2_dims, n_actions)

		self.optimizer = optim.Adam(self.parameters(), lr=lr)
		self.loss = nn.SmoothL1Loss()

	def forward(self, state):
		state = T.tensor(state, dtype=T.float32)
		out, (hn, cn) = self.fc1(state)
		out = out[:, -1, :]
		x = F.relu(self.fc2(out))
		return self.fc3(x)

class Agent():
	def __init__(self, input_dims, n_actions, batch_size=64, gamma=0.99, lr=0.00025, capacity=1000000, tau=1000):
		self.gamma = gamma
		self.tau = tau
		self.epsilon = 1
		self.step_counter = 0
		self.n_actions = n_actions

		self.replay_buffer = ReplayBufferSequence(capacity, input_dims, 3, batch_size)
		self.batch_size = batch_size

		self.Q_network = dqn_rnn(lr, input_dims, fc1_dims=128, fc2_dims=512, n_actions=n_actions)
		self.Q_target = dqn_rnn(lr, input_dims, fc1_dims=128, fc2_dims=512, n_actions=n_actions)
		self.Q_target.load_state_dict(self.Q_network.state_dict())
		self.Q_target.eval()

	def choose_action(self):
		if np.random.random() < self.epsilon:
			action = np.random.choice(self.n_actions)
		else:
			actions = self.Q_network.forward(self.replay_buffer.sequence)
			action = T.argmax(actions).item()
		return action

	def learn(self):
		if self.replay_buffer.size < self.batch_size:
			return
		self.Q_network.optimizer.zero_grad()

		batch_index, states, new_states, rewards, terminals, actions = self.replay_buffer.get_batch()

		test = self.Q_network(states)
		print('test', test)
		Q_values = test[batch_index, actions]
		next_actions = T.argmax(self.Q_network(new_states), dim=1)

		tmp = self.Q_target(new_states)
		next_Q_values = tmp[batch_index, next_actions]
		Q_targets = rewards + self.gamma * next_Q_values * (1 - terminals)

		loss = self.Q_network.loss(Q_targets, Q_values)
		loss.backward()
		self.Q_network.optimizer.step()

		if self.step_counter % self.tau == 0:
			self.Q_target.load_state_dict(self.Q_network.state_dict())
		self.step_counter += 1

	def save_model(self):
		T.save(self.Q_network.state_dict(), 'model.pth')

	def load_model(self):
		self.Q_network.load_state_dict(T.load('model.pth'))
		self.Q_target.load_state_dict(self.Q_network.state_dict())
