import numpy as np

class Agent:
	def __init__(self):
		self.nbr_states = 3 # Above, Bellow, In front
		self.nbr_actions = 3 # Up, Down, Stay
		self.goal_state = 2

		self.Q_table = np.zeros((self.nbr_states, self.nbr_actions))
		self.learning_rate = 0.8 # alpha
		self.discount_factor = 0.95 # gamma
		self.exploration_prob = 0.2 # epsilon

	def choose_action(self, current_state):
		if np.random.rand() < self.exploration_prob:
			return np.random.randint(0, self.nbr_actions)
		else:
			return np.argmax(self.Q_table[current_state])

	def update(self, current_state, action, reward, next_state):
		self.Q_table[current_state, action] += self.learning_rate * (reward + self.discount_factor * np.max(self.Q_table[next_state]) - self.Q_table[current_state, action])

	def get_reward(action, state):
		if action == state:
			return 1
		return -1

	def Qlearning(self, game):
		# Choose action
		current_state = game.get_paddle_state()
		action = self.choose_action(current_state)

		# Execute action
		if action == 0:
			game.paddle_left.move(0, -10)
		elif action == 1:
			game.paddle_left.move(0, 10)

		# Get a reward
		reward = self.get_reward(action, current_state)

		# Update Q-table
		current_state = game.get_paddle_state()
		self.update(current_state, action, reward, current_state)
