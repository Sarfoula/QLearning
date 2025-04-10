from pong import Game
from agent import Agent
import numpy as np
import signal
from time import sleep

class PongTrainer(Game):
	def __init__(self, height=600, width=800, delay=60, visual=False):
		super().__init__(height, width, visual)
		self.delay = delay
		self.input_dim = 6
		self.output_dim = 2
		self.capacity = 500000

		self.agent = Agent(self.input_dim, self.output_dim, capacity=self.capacity)

		self.running = True
		signal.signal(signal.SIGINT, self.signal_handler)
		signal.signal(signal.SIGTERM, self.signal_handler)

	def signal_handler(self, sig, frame):
		self.running = False

def train(env):
	for i in range(episode):
		state = env.reset()
		for frame in range(max_frame):
			if frame % delay == 0:
				actions = env.agent.get_actions(state)

			next_state, done = env.step(actions[frame % delay])

			if (frame + 1) % delay == 0:
				env.agent.store_transition(state, next_state)
				state = next_state

			loss = env.agent.learn()

			if loss is not None:
				total_loss.append(loss)

			if env.visual:
				env.root.update()

			if done:
				break

		if env.running == False:
			env.running = True
			break
		total_hit.append(env.paddle_left.hit)
		if i % 10 == 0 and len(total_loss) > 0:
			print(f"Ep {i}, Loss {np.mean(total_loss[-10:]):.6f}, hits {np.mean(total_hit[-10:]):.2f}")

def play(env):
	frame = 0
	state = env.reset()
	while(1):
		if frame % delay == 0:
			actions = env.agent.get_actions(state)
		action2 = env.get_key_action()

		next_state, done = env.step(actions[frame % delay], action2)
		state = next_state
		frame += 1
		if done:
			env.reset()
			frame = 0
		if env.visual:
			env.root.update()
		sleep(0.017)
		if env.running == False:
			break

if __name__ == "__main__":
	visual = False
	delay = 45
	episode = 1000
	total_loss = []
	total_hit = []
	max_frame = 4000

	env = PongTrainer(600, 800, delay, visual)
	train(env)
	env.agent.save_model("DELAY45.pt")

	# env.agent.load_model("DELAY45.pt")
	# env.agent.agent.eval()
	# play(env)
