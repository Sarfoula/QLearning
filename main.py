from pong import Game
from agent import *
from utils import *
import signal

loop = True
def stop(sig, frame):
	global loop
	loop = False

class trainGame(Game):
	def __init__(self, height, width, visual=False):
		super().__init__(height, width, visual)

		self.agent = Agent(6, 3, batch_size=64)

		self.visual = visual
		self.count_episode = 0
		self.reward = 0
		self.reward_episode = []

	def get_reward(self):
		reward = 0
		if self.ball_hit:
			self.ball_hit = False
			reward = 0.5
		if self.winner == 2:
			reward = -1
		elif self.winner == 1:
			reward = 1
		return reward

	def training(self, num_games=500):
		for i in range(num_games):
			if not loop:
				break

			self.time = 0
			self.reward = 0
			terminal = False
			self.reset()
			state = self.get_state()
			while not terminal:
				# choose action for both paddle
				left_action = self.agent.choose_action(state)
				right_action = self.opponent(self.paddle_right)

				# Step the game with actions and retrieve a bool terminal state
				terminal = self.step(left_action, right_action)

				# Get transition information and store it for ReplayBuffer
				reward = self.get_reward()
				new_state = self.get_state()
				self.agent.replay_buffer.store_transition(state, left_action, reward, new_state, terminal)
				state = new_state

				# Make the agent learn from a random batch of transition
				self.agent.learn()

				self.reward += reward
				if self.visual:
					self.root.update_idletasks()
					self.root.update()
			print('episode', i, 'reward', self.reward, 'hit', self.paddle_left.hit)
			print('epsilon', round(self.agent.epsilon, 3), '\n')
			self.reward_episode.append(self.reward)

if __name__ == "__main__":
	signal.signal(signal.SIGINT, stop)
	signal.signal(signal.SIGTERM, stop)

	episodes = 2000
	game = trainGame(600, 800, visual=False)
	game.training(episodes)
	show_result(game.reward_episode)

