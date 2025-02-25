import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeepQNetwork(nn.Module):
	def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
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

class ReplayBuffer():
	def __init__(self, max_size, input_dims, batch_size=64, reward_factor=0.05):
		self.max_size = max_size
		self.batch_size = batch_size
		self.prev_actions = []
		self.reward_factor = reward_factor
		self.mem_number_element = 0

		self.state_memory = np.zeros((self.max_size, input_dims), dtype=np.float32)
		self.new_state_memory = np.zeros((self.max_size, input_dims), dtype=np.float32)
		self.action_memory = np.zeros(self.max_size, dtype=np.int32)
		self.reward_memory = np.zeros(self.max_size, dtype=np.float32)
		self.terminal_memory = np.zeros(self.max_size, dtype=np.uint8)

	def store_transition(self, state, action, reward, state_, done, ball_hit):
		self.prev_actions.append((state, state_, reward, action, done))

		if ball_hit or done:
			for i, (s, s_, r, a, d) in enumerate(self.prev_actions):
				index = self.mem_number_element % self.max_size
				self.state_memory[index] = s
				self.new_state_memory[index] = s_
				self.reward_memory[index] = r + reward * self.reward_factor * (i + 1)
				self.action_memory[index] = a
				self.terminal_memory[index] = int(d)
				self.mem_number_element += 1
			self.prev_actions = []

	def get_batch(self):
		max_mem = min(self.mem_number_element, self.max_size)
		batch = np.random.choice(max_mem, self.batch_size, replace=False)
		batch_index = np.arange(self.batch_size, dtype=np.int32)

		state_batch = T.tensor(self.state_memory[batch], dtype=T.float)
		new_state_batch = T.tensor(self.new_state_memory[batch], dtype=T.float)
		reward_batch = T.tensor(self.reward_memory[batch], dtype=T.float)
		terminal_batch = T.tensor(self.terminal_memory[batch], dtype=T.float)
		action_batch = T.tensor(self.action_memory[batch], dtype=T.int64)

		return batch_index, state_batch, new_state_batch, reward_batch, terminal_batch, action_batch

class Agent():
	def __init__(self, input_dims, n_actions, batch_size=128, gamma=0.99, lr=0.0005, max_size=1000000, tau=1000, reward_factor=0.05):
		self.gamma = gamma
		self.tau = tau
		self.epsilon = 1
		self.step_counter = 0
		self.n_actions = n_actions

		self.replayBuffer = ReplayBuffer(max_size, input_dims, batch_size, reward_factor)

		self.Q_network = DeepQNetwork(lr, input_dims, fc1_dims=256, fc2_dims=256, n_actions=n_actions)
		self.Q_target = DeepQNetwork(lr, input_dims, fc1_dims=256, fc2_dims=256, n_actions=n_actions)
		self.Q_target.load_state_dict(self.Q_network.state_dict())
		self.Q_target.eval()

	def choose_action(self, state):
		if np.random.random() < self.epsilon:
			action = np.random.choice(self.n_actions)
		else:
			state = T.tensor(state, dtype=T.float)
			actions = self.Q_network(state)
			action = T.argmax(actions).item()
		return action

	def learn(self):
		if self.replayBuffer.mem_number_element < self.replayBuffer.batch_size:
			return
		self.Q_network.optimizer.zero_grad()

		batch_index, states, new_states, rewards, terminals, actions = self.replayBuffer.get_batch()

		test = self.Q_network(states)
		Q_values = T.gather(test, 1, actions.unsqueeze(1)).squeeze(1)
		# Q_values = test[batch_index, actions]
		next_action = T.argmax(self.Q_network(new_states), dim=1)
		next_Q_values = self.Q_target(new_states)[batch_index, next_action]
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
