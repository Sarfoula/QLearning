import torch as T
import torch.nn as nn
import torch.optim as optim
import random

class ReplayBuffer:
	def __init__(self, buffer_size, batch_size):
		self.batch_size = batch_size
		self.buffer_size = buffer_size
		self.buffer = []
		self.size = 0

	def store_transition(self, state, action, reward, next_state, done):
		if self.size >= self.buffer_size:
			self.buffer.pop(0)
		self.buffer.append((state, action, reward, next_state, done))
		self.size += 1

	def get_batch(self):
		batch = random.sample(self.buffer, self.batch_size)

		states, actions, rewards, next_states, dones = zip(*batch)

		states = T.FloatTensor(states)
		next_states = T.FloatTensor(next_states)
		actions = T.LongTensor(actions).unsqueeze(1)
		rewards = T.FloatTensor(rewards)
		dones = T.FloatTensor(dones)

		return states, actions, rewards, next_states, dones

class DeepQNetwork(nn.Module):
	def __init__(self, input_dim, output_dim, fc1_dims=64, fc2_dims=64):
		super(DeepQNetwork, self).__init__()
		self.fc1 = nn.Linear(input_dim, fc1_dims)
		self.fc2 = nn.Linear(fc1_dims, fc2_dims)
		self.fc3 = nn.Linear(fc2_dims, output_dim)

		self.loss = nn.SmoothL1Loss()
		self.optimizer = optim.Adam(self.parameters(), lr=0.001)

	def forward(self, state):
		x = T.relu(self.fc1(state))
		x = T.relu(self.fc2(x))
		return self.fc3(x)

class Agent():
	def __init__(self, input_dim, output_dim, batch_size=64):
		# discount factor for the Bellman equation
		self.gamma = 0.99

		# Number of step for update target network
		self.tau = 1000
		self.step_counter = 0

		# epsilon greedy policy for exploration, decrease over time by deacay_rate and never goes below min
		self.epsilon = 1
		self.epsilon_min = 0.01
		self.decay_rate = 0.99997

		# Size for batch sample to learn from
		self.replay_buffer = ReplayBuffer(10000, batch_size)

		# 2 neural network for the actual policy and for the target
		self.policy = DeepQNetwork(input_dim, output_dim, fc1_dims=64, fc2_dims=64)
		self.target = DeepQNetwork(input_dim, output_dim, fc1_dims=64, fc2_dims=64)
		self.target.load_state_dict(self.policy.state_dict())
		self.target.eval()

	def epsilon_decay(self):
		"""Decrease the epsilon exponentioly by a certain decay rate"""
		self.epsilon = max(self.epsilon_min, self.epsilon * self.decay_rate)

	def choose_action(self, state):
		"""Choose action """
		if random.random() < self.epsilon:
			action = random.randint(0, 2)
		else:
			with T.no_grad():
				action = T.argmax(self.policy.forward(T.FloatTensor(state))).item()
		self.epsilon_decay()
		return action

	def learn(self):
		"""Make the agent learn from a batch_size experience"""
		if self.replay_buffer.size < self.replay_buffer.batch_size:
			return

		# Gives a random batch transition
		states, actions, rewards, new_states, terminals = self.replay_buffer.get_batch()

		# Get q_value for state with the action choosen
		q_values = self.policy(states)
		q_value = T.gather(q_values, 1, actions).squeeze(1)

		# Get best next_q_value for next_state with best next_action
		next_max_action = T.argmax(self.policy(new_states), 1, True)
		max_next_q_value = T.gather(self.target(new_states), 1, next_max_action).squeeze(1)

		# Bellman equation gives use the target q_value (the ideal q_value)
		q_target = rewards + self.gamma * max_next_q_value * (1 - terminals)

		# The loss function is a difference between q_value and the q_target
		self.policy.optimizer.zero_grad()
		loss = self.policy.loss(q_value, q_target)

		# Change the weight in the neural network to fit the q_target
		loss.backward()
		self.policy.optimizer.step()

		# Update the target network every tau step
		if self.step_counter % self.tau == 0:
			self.target.load_state_dict(self.policy.state_dict())
		self.step_counter += 1

	def save_model(self):
		T.save(self.policy.state_dict(), 'model.pth')

	def load_model(self):
		self.policy.load_state_dict(T.load('model.pth'))
		self.target.load_state_dict(self.policy.state_dict())
