import torch as T
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class StatePredictorNetwork(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_size=128):
		super(StatePredictorNetwork, self).__init__()
		self.encoder = nn.Sequential(nn.Linear(state_dim+action_dim, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size))
		self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
		self.residual = nn.Linear(state_dim, hidden_size)
		self.decoder = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, state_dim))

	def forward(self, state, action, hidden=None):
		tensor = T.cat([state, action], dim=1)

		encoded = self.encoder(tensor).unsqueeze(1)
		if hidden is None:
			lstm_out, hidden = self.lstm(encoded)
		else:
			lstm_out, hidden = self.lstm(encoded, hidden)
		lstm_out = lstm_out.squeeze(1)
		residual_connection = self.residual(state)

		combined = lstm_out + residual_connection
		next_state = self.decoder(combined)

		return next_state, hidden

class StatePredictor():
	def __init__(self, state_dim, action_dim, sequence_len, lr=0.005, batch_size=64, hidden_size=128, capacity=100000):
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.sequence_len = sequence_len
		self.batch_size = batch_size
		self.capacity = capacity

		self.buffer = []
		self.curr_sequence = []

		self.predictor = StatePredictorNetwork(state_dim, action_dim, hidden_size)
		self.loss = nn.MSELoss()
		self.optim = optim.Adam(self.predictor.parameters(), lr=lr)

	def predict_state(self, state, action, hidden=None):
		one_hot = T.zeros(1, self.action_dim, dtype=T.float32)
		one_hot[0, action] = 1.0

		with T.no_grad():
			prediction, hidden = self.predictor(state.unsqueeze(0), one_hot, hidden)

		return prediction.squeeze(0).detach(), hidden

	def store_transition(self, state, action, next_state=None):
		if next_state is None:
			self.curr_sequence.append((state, action))
		else:
			self.curr_sequence.append((state, action, next_state))

		if len(self.curr_sequence) >= self.sequence_len:
			self.buffer.append(list(self.curr_sequence))
			self.curr_sequence = []

			if len(self.buffer) > self.capacity:
				self.buffer.pop()

	def init_batch(self, batch_seq):
		batch_state = []
		batch_action = []
		batch_next_state = []

		for sequence in batch_seq:
			seq_state = []
			seq_action = []
			seq_next_state = []

			for i in range(len(sequence) - 1):
				state, action, next_state = sequence[i]

				seq_state.append(state)

				one_hot_action = T.zeros(self.action_dim, dtype=T.float32)
				one_hot_action[action] = 1.0
				seq_action.append(one_hot_action)

				seq_next_state.append(next_state)

			batch_state.append(seq_state)
			batch_action.append(seq_action)
			batch_next_state.append(seq_next_state)

		states = T.tensor(np.array(batch_state), dtype=T.float32)
		actions = T.tensor(np.array(batch_action), dtype=T.float32)
		next_states = T.tensor(np.array(batch_next_state), dtype=T.float32)

		return states, actions, next_states

	def learn(self, epoch):
		if len(self.buffer) < self.batch_size:
			return

		losses = []
		for i in range(epoch):
			epoch_loss = []
			for j in range(len(self.buffer) // self.batch_size):
				batch = random.sample(self.buffer, self.batch_size)
				states, actions, next_states = self.init_batch(batch)

				_, seq_len, _ = states.shape

				self.predictor.train()
				self.optim.zero_grad()

				hidden = None
				total_loss = 0

				for k in range(seq_len):
					state = states[:, k, :]
					action = actions[:, k, :]
					target = next_states[:, k, :]

					prediction, hidden = self.predictor(state, action, hidden)

					loss = self.loss(prediction, target)
					total_loss += loss

				total_loss = total_loss / seq_len
				total_loss.backward()
				T.nn.utils.clip_grad_norm_(self.predictor.parameters(), max_norm=1.0)
				self.optim.step()

				epoch_loss.append(total_loss.item())

			avg_loss = sum(epoch_loss) / len(epoch_loss)
			losses.append(avg_loss)
			print(f"Epoch {i}/{epoch}, loss: {avg_loss:.6f}")

		return losses

	def save_model(self, path):
		T.save({'model_state_dict': self.predictor.state_dict(), 'optimizer_state_dict': self.optim.state_dict(),}, path)

	def load_model(self, path):
		checkpoint = T.load(path)
		self.predictor.load_state_dict(checkpoint['model_state_dict'])
		self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
