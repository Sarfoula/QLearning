from pong import Game
from agent import *
from utils import show_result
from time import sleep
import signal

class trainGame(Game):
	def __init__(self, height, width, visual=False):
		super().__init__(height, width, visual)

		self.visual = visual
		self.capacity = 500000
		self.agent = Agent(5, 3, batch_size=64, capacity=self.capacity)

	def ai(self, state):
		with T.no_grad():
			return T.argmax(self.agent.policy.forward(T.tensor(state, dtype=T.float32).unsqueeze(0))).item()

	def training(self, num_games=500):
		flag = 0
		max_frame = 4000
		history_reward, history_loss = [], []
		for i in range(num_games):
			terminal = False
			state = self.reset()
			rewards = 0
			losses = 0
			for frame in range(max_frame):
				left_action = self.agent.choose_action(state)
				right_action = self.opponent(self.paddle_right)
				new_state, reward, terminal = self.step(left_action, right_action)

				self.agent.replay_buffer.store_transition(state, left_action, reward, new_state, terminal)
				loss = self.agent.learn()

				state = new_state
				rewards += reward
				if loss is not None:
					losses += loss
				if terminal:
					break
			self.agent.epsilon_decay()
			if self.agent.replay_buffer.tree.size == self.capacity and flag == 0:
				print(f"max capacity reached {i}")
				flag = i
			history_loss.append(losses)
			history_reward.append(rewards)
			if i % 10 == 0:
				print(f"Ep {i} /{num_games} Rwrd {np.mean(history_reward):.2f} Eps {self.agent.epsilon:.3f}")
			if not loop:
				break

		return history_reward, history_loss, str(i), flag

	def play(self):
		self.agent.policy.eval()
		while (1):
			action1 = self.ai(self.get_state())
			action2 = self.get_key_action()
			_, reward, done = self.step(action1, action2)

			if reward:
				print(reward)
			if done:
				self.reset()
			self.root.update_idletasks()
			self.root.update()
			sleep(0.017)

if __name__ == "__main__":
	loop = True
	def stop(sig, frame):
		global loop
		loop = False
	signal.signal(signal.SIGINT, stop)
	signal.signal(signal.SIGTERM, stop)

	name = "LongTraining"
	episodes = 5000

	game = trainGame(600, 800, visual=True)
	game.agent.load_model("models/LongTraining834.pt")
	game.play()
	# score, loss, episode, flag = game.training(episodes)
	# show_result(score, loss, name)
	# game.agent.save_model("models/" + name + episode + ".pt")
