from pong import Game
from agent import *
from utils import *
import signal

class trainGame(Game):
	def __init__(self, height, width, delay, stop, visual=False):
		super().__init__(height, width, visual)

		self.agent = Agent(6, 3, batch_size=64)

		self.frame = 0
		self.delay = delay
		self.visual = visual
		self.reward = 0

		self.reward_not_improve = 0
		self.reward_stop = stop
		self.last_reward = 0

		self.epsilon_update = 0
		self.ball_hit_episode = []
		self.distance_error = []
		self.reward_episode = []

	def get_reward(self):
		reward = 0
		if self.ball_hit:
			self.ball_hit = False
			reward = 1

		if self.winner == 2:
			dist = abs((self.ball.y - self.paddle_left.y) / self.height)
			self.distance_error.append(dist)
			reward = -dist
		return reward

	def get_state(self):
		"""return paddle_y, ball_x, ball_y, ball_vx, ball_vy, timer"""
		return np.array([self.paddle_left.y/self.height,
				self.ball.x/self.width,
				self.ball.y/self.height,
				self.ball.vx/self.ballspeed,
				self.ball.vy/self.ballspeed,
				self.frame % self.delay], dtype=np.float32)

	def training(self, num_games=500):
		for i in range(num_games):
			if not loop or self.reward_not_improve > self.reward_stop:
				break

			self.frame = 0
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
				self.frame += 1

				# Get transition information and store it for ReplayBuffer
				reward = self.get_reward()
				if self.frame % self.delay == 0 or terminal:
					new_state = self.get_state()
				else:
					new_state = state.copy()
					new_state[5] == self.frame % self.delay

				self.agent.replay_buffer.store_transition(state, left_action, reward, new_state, terminal)
				state = new_state

				# Make the agent learn from a random batch of transition
				self.agent.learn()

				self.reward += reward
				if self.visual:
					self.root.update_idletasks()
					self.root.update()

			self.reward_episode.append(self.reward)
			self.ball_hit_episode.append(self.paddle_left.hit)

			if self.reward < self.last_reward:
				self.reward_not_improve += 1
			else:
				self.reward_not_improve = 0

			self.last_reward = self.reward

			if self.agent.epsilon == self.agent.epsilon_min and self.epsilon_update == 0:
				self.epsilon_update = i
			print(f'episode {i}, mean reward {np.mean(self.reward_episode):.2f}, mean hit {self.paddle_left.hit}')
			print(f'epsilon {self.agent.epsilon:.3f}\n')

	def play(self):
		left_action = self.agent.real_choose(self.state)
		right_action = self.get_key_action()

		self.step(left_action, right_action)
		self.frame += 1

		if self.frame % self.delay == 0:
			self.state = self.get_state()
		else:
			self.state[5] += 1

		if self.winner != 0:
			self.reset()
			self.state = self.get_state()
		self.root.after(17, self.play)

if __name__ == "__main__":
	loop = True
	def stop(sig, frame):
		global loop
		loop = False
	signal.signal(signal.SIGINT, stop)
	signal.signal(signal.SIGTERM, stop)

	delay = 60
	episodes = 5000

	game = trainGame(600, 800, visual=None, delay=delay, stop=10)
	name = "origin"
	game.training(episodes)
	show_result(game.reward_episode,
			 game.ball_hit_episode,
			 game.distance_error,
			 game.epsilon_update,
			 name + ".png")
