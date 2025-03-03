import numpy as np
import torch as T
import random

class SumTree:
	def __init__(self, capacity):
		self.capacity = capacity
		self.tree = np.zeros(2 * capacity - 1)
		self.data = {
			'state': np.zeros((capacity, 6), dtype=np.float32),
			'next_state': np.zeros((capacity, 6), dtype=np.float32),
			'action': np.zeros(capacity, dtype=np.int32),
			'reward': np.zeros(capacity, dtype=np.float32),
			'done': np.zeros(capacity, dtype=np.uint8),
		}
		self.ptr = 0
		self.size = 0

	def propagate(self, idx, change):
		"""Met à jour les valeurs dans l'arbre après modification."""
		parent = (idx - 1) // 2
		self.tree[parent] += change
		if parent != 0:
			self.propagate(parent, change)

	def retrieve(self, idx, sum_):
		"""Trouve l'indice de l'échantillon avec une probabilité basée sur la somme."""
		left = 2 * idx + 1
		right = left + 1

		if left >= len(self.tree):
			return idx - self.capacity + 1

		if sum_ <= self.tree[left]:
			return self.retrieve(left, sum_)
		else:
			return self.retrieve(right, sum_ - self.tree[left])

	def add(self, priority, data):
		"""Ajoute une donnée avec une priorité."""
		idx = self.pointer + self.capacity - 1
		self.data[self.pointer] = data
		self.update(idx, priority)

		self.pointer += 1
		if self.pointer >= self.capacity:
			self.pointer = 0

	def update(self, idx, priority):
		"""Met à jour la priorité d'une transition."""
		change = priority - self.tree[idx]
		self.tree[idx] = priority
		self.propagate(idx, change)

	def get(self, priority_sum):
		"""Récupère un échantillon avec une probabilité basée sur la priorité."""
		idx = self._retrieve(0, priority_sum)
		data = self.data[idx]
		return idx, data

	def total_priority(self):
		"""Retourne la somme totale des priorités."""
		return self.tree[0]

class PrioritizedReplayBuffer:
	def __init__(self, capacity, alpha=0.6, beta=0.4):
		self.size = 0
		self.alpha = alpha
		self.beta = beta
		self.tree = SumTree(capacity)

	def add(self, error, transition):
		priority = (error + 1e-5) ** self.alpha
		self.tree.add(priority, transition)

	def sample(self, batch_size):
		batch = []
		idxs = []
		priorities = []

		segment = self.tree.total_priority() / batch_size
		for i in range(batch_size):
			s = random.uniform(i * segment, (i + 1) * segment)
			idx, data = self.tree.get(s)
			batch.append(data)
			idxs.append(idx)
			priorities.append(self.tree.tree[idx + self.capacity - 1])

		total_priority = self.tree.total_priority()
		weights = np.array([((total_priority * priority) ** (-self.beta)) for priority in priorities])
		weights /= weights.max()

		return batch, idxs, weights

	def update_priorities(self, idxs, errors):
		priorities = (np.abs(errors) + self.epsilon) ** self.alpha
		for idx, priority in zip(idxs, priorities):
			self.tree.update(idx, priority)

class NewPersonalType():
	def __init__(self, state, next_state, capacity):
		self.capacity = capacity
		self.sequence = np.full((capacity, len(state)), state, dtype=np.float32)
		self.next_sequence = np.full((capacity, len(next_state)), next_state, dtype=np.float32)

	def add(self, state, next_state):
		for i in range(self.capacity):
			self.sequence[i] = self.sequence[i + 1]
			self.next_sequence[i] = self.next_sequence[i + 1]
		self.sequence[self.capacity - 1] = state
		self.next_sequence[self.capacity - 1] = next_state

class ReplayBufferSequence():
	def __init__(self, capacity, input_dims, sequence_size, batch_size=64):
		self.capacity = capacity
		self.sequence_size = sequence_size
		self.batch_size = batch_size
		self.states = np.zeros((capacity, sequence_size, input_dims), dtype=np.float32)
		self.new_states = np.zeros((capacity, sequence_size, input_dims), dtype=np.float32)
		self.reward = np.zeros(capacity, dtype=np.float32)
		self.action = np.zeros(capacity, dtype=np.int32)
		self.done = np.zeros(capacity, dtype=np.uint8)
		self.size = 0

		self.sequence = np.zeros((sequence_size, input_dims), dtype=np.float32)
		self.next_sequence = np.zeros((sequence_size, input_dims), dtype=np.float32)

	def add(self, state, next_state):
		for i in range(self.sequence_size - 1):
			self.sequence[i] = self.sequence[i + 1]
			self.next_sequence[i] = self.next_sequence[i + 1]
		self.sequence[self.sequence_size - 1] = state
		self.next_sequence[self.sequence_size - 1] = next_state

	def store_transition(self, state, action, reward, next_state, done):
		self.add(state, next_state)

		index = self.size % self.capacity
		self.states[index] = state
		self.new_states[index] = next_state
		self.reward[index] = reward
		self.action[index] = action
		self.done[index] = int(done)
		self.size += 1

	def get_batch(self):
		max_mem = min(self.size, self.capacity)
		batch = np.random.choice(max_mem, self.batch_size, replace=False)
		batch_index = np.arange(self.batch_size, dtype=np.int32)

		state_batch = T.tensor(self.states[batch], dtype=T.float)
		new_state_batch = T.tensor(self.new_states[batch], dtype=T.float)
		reward_batch = T.tensor(self.reward[batch], dtype=T.float)
		done_batch = T.tensor(self.done[batch], dtype=T.float)
		action_batch = T.tensor(self.action[batch], dtype=T.int64)

		return batch_index, state_batch, new_state_batch, reward_batch, done_batch, action_batch

