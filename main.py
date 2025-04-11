import time
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import random

class SumTree:
	def __init__(self, capacity):
		self.capacity = capacity
		self.tree = np.zeros(2 * capacity - 1)
		self.data = np.zeros(capacity, dtype=object)
		self.size = 0
		self.ptr = 0
		self.counter = 0

	def add(self, priority, sample):
		idx = self.ptr + self.capacity - 1
		self.data[self.ptr] = sample
		self.update(idx, priority)

		self.ptr = (self.ptr + 1) % self.capacity
		self.size = min(self.size + 1, self.capacity)
		self.counter += 1

	def update(self, idx, priority):
		change = priority - self.tree[idx]
		self.tree[idx] = priority
		while idx != 0:
			idx = (idx - 1) // 2
			self.tree[idx] += change

	def get_leaf(self, idx):
		data_idx = idx - self.capacity + 1
		return idx, self.tree[idx], self.data[data_idx]

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
		return self.get_leaf(idx)

	def total_priority(self):
		return self.tree[0]

	def __len__(self):
		return self.size

class PrioritizedReplayBuffer:
	def __init__(self, capacity, batch_size, alpha=0.6, beta=0.4, beta_inc=0.001, epsilon=0.01):
		self.tree = SumTree(capacity)
		self.batch_size = batch_size
		self.alpha = alpha
		self.beta = beta
		self.beta_inc = beta_inc
		self.epsilon = epsilon
		self.max_priority = 1.0
		self.last_indices = None

	def store(self, state, target):
		priority = self.max_priority
		if isinstance(state, T.Tensor):
			state = state.detach().cpu().numpy()
		if isinstance(target, T.Tensor):
			target = target.detach().cpu().numpy()
		self.tree.add(priority, (state, target))

	def sample(self):
		if len(self.tree) < self.batch_size:
			return None

		batch_indices = []
		batch_states = []
		batch_targets = []
		batch_weights = []

		total_priority = self.tree.total_priority()
		if total_priority <= 0:
			return None

		self.beta = min(1.0, self.beta + self.beta_inc)
		segment = total_priority / self.batch_size

		for i in range(self.batch_size):
			a, b = segment * i, segment * (i + 1)
			s = random.uniform(a, b)

			idx, priority, data = self.tree.sample(s)

			sampling_probability = priority / total_priority
			weight = (sampling_probability * len(self.tree)) ** -self.beta

			batch_indices.append(idx)
			state, target = data
			batch_states.append(T.tensor(state, dtype=T.float32))
			batch_targets.append(T.tensor(target, dtype=T.float32))
			batch_weights.append(weight)

		batch_weights = np.array(batch_weights) / max(batch_weights)

		states = T.stack(batch_states)
		targets = T.tensor(batch_targets, dtype=T.float32).unsqueeze(1)
		weights = T.tensor(batch_weights, dtype=T.float32)

		self.last_indices = batch_indices

		return states, targets, weights

	def update_priorities(self, td_errors):
		if self.last_indices is None:
			return

		if isinstance(td_errors, T.Tensor):
			td_errors = td_errors.detach().cpu().numpy()

		for idx, error in zip(self.last_indices, td_errors):
			priority = (abs(error) + self.epsilon) ** self.alpha
			self.tree.update(idx, priority)
			self.max_priority = max(self.max_priority, priority)

		self.last_indices = None

	def __len__(self):
		return len(self.tree)

class Network(nn.Module):
	def __init__(self, input_dim, output_dim, fc1_dims=64, fc2_dims=64):
		super().__init__()
		self.network = nn.Sequential(
		nn.Linear(input_dim, fc1_dims),
		nn.ReLU(),
		nn.Linear(fc1_dims, fc2_dims),
		nn.ReLU(),
		nn.Linear(fc2_dims, output_dim))

	def forward(self, state):
		return self.network(state)

class PongAI:
	def __init__(self):
		self.predictor = Network(7, 1)
		self.batch_size = 64

		self.actions = []
		self.paddle_speed = 10
		self.buffer = PrioritizedReplayBuffer(10000, self.batch_size)

		self.sequence = []
		self.loss = nn.HuberLoss(reduction='none')
		self.optimizer = optim.Adam(self.predictor.parameters(), lr=0.0003)

	def save_model(self, path="save.pt"):
		T.save({'model_state_dict': self.predictor.state_dict(),
		  'optim_state_dict': self.optimizer.state_dict()}, path)

	def load_model(self, path="save.pt"):
		file = T.load(path)
		self.predictor.load_state_dict(file['model_state_dict'])
		self.optimizer.load_state_dict(file['optim_state_dict'])
		self.predictor.eval()

	def store_transition(self):
		target = self.sequence[-1][3]
		for state in self.sequence:
			self.buffer.store(state, target)
		self.sequence = []

	def store_sequence(self, state):
		self.sequence.append(state)

	def learn(self):
		if len(self.buffer) < self.batch_size:
			return None

		result = self.buffer.sample()
		if result is None:
			return None

		states, targets, weights = result
		predictions = self.predictor(states)

		with T.no_grad():
			td_errors = T.abs(predictions - targets).squeeze().cpu().numpy()

		elementwise_loss = self.loss(predictions, targets)
		loss = (elementwise_loss * weights.unsqueeze(1)).mean()

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		self.buffer.update_priorities(td_errors)

		return loss.item()

	def predict_y(self, state):
		with T.no_grad():
			prediction = self.predictor(T.tensor(state, dtype=T.float32)).item()
		return prediction

	def train(self, epoch):
		delay = 60
		state = game.reset()
		last_direction = state[4]
		losses = []
		distance = []

		for frame in range(epoch):
			if frame % delay == 0:
				predict = self.predict_y(state)
			if state[4] * last_direction > 0 and game.paddle_left.x + 5 < state[2] < game.paddle_right.x - 5:
				self.store_sequence(state)
			else:
				self.store_transition()
				last_direction = state[4]
				distance.append(abs(predict - state[3]))

			limit = None
			action1 = game.opponent(game.paddle_left, limit)
			action2 = game.opponent(game.paddle_right, limit)
			state, done = game.step(action1, action2)

			loss = self.learn()
			if loss is not None:
				losses.append(loss)

			if frame % 100 == 0 and loss:
				print(f"{frame}, loss {np.mean(losses[-10:]):.3f}, distance {np.mean(distance[-10:]):.0f}")

			if done:
				print("				LOSSETETETETTE")
				state = game.reset()
				last_direction = state[4]
			if game.visual:
				game.root.update()

	def update(self, paddle_y, predict_y):
		distance = paddle_y - predict_y

		if distance < 0:
			action = 1
		elif distance > 0:
			action = 0

		distance = abs(distance) // self.paddle_speed

		self.actions = []
		for i in range(int(distance)):
			self.actions.append(action)

	def get_action(self):
		if len(self.actions) > 0:
			return self.actions.pop(0)
		else:
			return 2

	def test(self):
		frame = 0
		delay = 60
		state = game.reset()

		while(1):
			if frame % delay == 0:
				if state[4] > 0:
					predict = self.predict_y(state)
					self.update(state[1], predict)
					game.prediction.tp(game.paddle_right.x, predict)

			action1 = game.get_key_action()

			action2 = self.get_action()
			state, done = game.step(action1, action2)
			frame += 1

			if done:
				state = game.reset()
				frame = 0
			if game.visual:
				game.root.update()
			time.sleep(0.017)


if __name__ == "__main__":
	from pong import Game

	game = Game(height=600, width=800, visual=False)
	ai = PongAI()

	ai.train(100000)
	ai.save_model()

	ai.load_model()
	ai.test()
