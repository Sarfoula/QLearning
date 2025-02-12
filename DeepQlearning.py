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
		self.loss = nn.SmoothL1Loss() # Can use Huber loss ?
		self.device = T.device('cpu')
		self.to(self.device)

	def forward(self, state):
		x = F.relu(self.fc1(state))
		x = F.relu(self.fc2(x))
		return self.fc3(x)

class Agent():
	def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, mem_size=100000, eps_end=0.01, eps_decay=0.995, tau=0.01):
		self.gamma = gamma
		self.epsilon = epsilon
		self.eps_start = epsilon
		self.eps_decay = eps_decay
		self.eps_end = eps_end
		self.tau = tau

		self.n_actions = n_actions
		self.mem_size = mem_size
		self.batch_size = batch_size
		self.mem_cntr = 0

		self.Q_eval = DeepQNetwork(lr, input_dims, fc1_dims=256, fc2_dims=256, n_actions=n_actions)
		self.Q_target = DeepQNetwork(lr, input_dims, fc1_dims=256, fc2_dims=256, n_actions=n_actions)
		self.Q_target.load_state_dict(self.Q_eval.state_dict())
		self.Q_target.eval()

		# Memory replay
		self.state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
		self.new_state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
		self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
		self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
		self.terminal_memory = np.zeros(self.mem_size, dtype=np.uint8)

	def store_transition(self, state, action, reward, state_, done):
		index = self.mem_cntr % self.mem_size
		self.state_memory[index] = state
		self.new_state_memory[index] = state_
		self.reward_memory[index] = reward
		self.action_memory[index] = action
		self.terminal_memory[index] = int(done)

		self.mem_cntr += 1

	def choose_action(self, state):
		if np.random.random() > self.epsilon:
			state = T.tensor(state, dtype=T.float).to(self.Q_eval.device)
			Qvalues = self.Q_eval(state)
			action = T.argmax(Qvalues).item()
		else:
			action = np.random.choice(self.n_actions)

		return action

	def learn(self):
		if self.mem_cntr < self.batch_size:
			return

		self.Q_eval.optimizer.zero_grad()

		max_mem = min(self.mem_cntr, self.mem_size)
		batch = np.random.choice(max_mem, self.batch_size, replace=False)

		batch_index = np.arange(self.batch_size, dtype=np.int32)

		state_batch = T.tensor(self.state_memory[batch], dtype=T.float).to(self.Q_eval.device)
		new_state_batch = T.tensor(self.new_state_memory[batch], dtype=T.float).to(self.Q_eval.device)
		reward_batch = T.tensor(self.reward_memory[batch], dtype=T.float).to(self.Q_eval.device)
		terminal_batch = T.tensor(self.terminal_memory[batch], dtype=T.float).to(self.Q_eval.device)

		action_batch = self.action_memory[batch]

		q_eval = self.Q_eval(state_batch)[batch_index, action_batch]
		q_next = self.Q_target(new_state_batch).detach()
		q_next_max = T.max(q_next, dim=1)[0]
		q_target = reward_batch + self.gamma * q_next_max * (1 - terminal_batch)

		loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
		loss.backward()
		self.Q_eval.optimizer.step()

		for param, target_param in zip(self.Q_eval.parameters(), self.Q_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		# self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)

	def save_model(self):
		T.save(self.Q_eval.state_dict(), 'model.pth')

	def load_model(self):
		self.Q_eval.load_state_dict(T.load('model.pth'))
		self.Q_target.load_state_dict(self.Q_eval.state_dict())
