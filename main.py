from pong import Game
from agent import *
import signal

class trainGame(Game):
	def __init__(self, height, width, visual=False):
		super().__init__(height, width, visual)

		self.visual = visual
		self.agent = Agent(5, 3, batch_size=64)

	def training(self, num_games=500):
		for i in range(num_games):
			terminal = False
			state = self.reset()

			while not terminal:
				left_action = self.agent.choose_action(state)
				right_action = self.opponent(self.paddle_right)
				new_state, reward, terminal = self.step(left_action, right_action)

				self.agent.replay_buffer.store_transition(state, left_action, reward, new_state, terminal)
				self.agent.learn()

				state = new_state
				if self.visual:
					self.root.update_idletasks()
					self.root.update()
			if not loop:
				break

if __name__ == "__main__":
	loop = True
	def stop(sig, frame):
		global loop
		loop = False
	signal.signal(signal.SIGINT, stop)
	signal.signal(signal.SIGTERM, stop)

	episodes = 5000

	game = trainGame(600, 800, visual=False)
	game.training(episodes)
