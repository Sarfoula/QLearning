from pong import Game
from agent import Agent
from utils import show_loss, show_result
from StatePredictor import StatePredictor
import torch as T
import numpy as np
import os
import pickle
import signal
import random
from time import sleep

class PongTrainer(Game):
	def __init__(self, height=600, width=800, delay=60, visual=False):
		super().__init__(height, width, visual)
		self.save_dir = "models/"
		self.delay = delay
		self.input_dim = 5
		self.output_dim = 3
		self.capacity = 500000

		self.agent = Agent(self.input_dim, self.output_dim, batch_size=64, capacity=self.capacity)
		self.predictor = StatePredictor(self.input_dim, self.output_dim, delay,
									   lr=0.005, batch_size=64, hidden_size=128)

		self.episode_rewards = []
		self.agent_losses = []
		self.predictor_losses = []

		self.running = True
		signal.signal(signal.SIGINT, self.signal_handler)
		signal.signal(signal.SIGTERM, self.signal_handler)

		os.makedirs(self.save_dir, exist_ok=True)

	def signal_handler(self, sig, frame):
		self.running = False

	def save_models(self, suffix=""):
		self.agent.save_model(os.path.join(self.save_dir, f"agent_{suffix}.pt"))
		self.predictor.save_model(os.path.join(self.save_dir, f"predictor_{suffix}.pt"))

	def load_models(self, suffix=""):
		agent_path = os.path.join(self.save_dir, f"agent_{suffix}.pt")
		if os.path.exists(agent_path):
			self.agent.load_model(agent_path)

		predictor_path = os.path.join(self.save_dir, f"predictor_{suffix}.pt")
		if os.path.exists(predictor_path):
			self.predictor.load_model(predictor_path)

	def train_predictor(self, num_episodes=100, frames_per_episode=4000):
		for episode in range(1, num_episodes + 1):
			if not self.running:
				break

			state = self.reset()
			states = [state]
			actions = []

			for frame in range(frames_per_episode):
				if random.random() < 0.2:
					action = random.randint(0, self.output_dim-1)
				else:
					action = self.opponent(self.paddle_left)
				actions.append(action)

				next_state, reward, terminal = self.step(action)
				states.append(next_state)
				state = next_state

				if self.visual:
					self.root.update_idletasks()
					self.root.update()
				if terminal:
					break

			for i in range(len(states) - 1):
				self.predictor.store_transition(states[i], actions[i], states[i+1])

			if episode % 5 == 0:
				losses = self.predictor.learn(epoch=3)
				if losses:
					self.predictor_losses.extend(losses)
					print(f"Épisode {episode}/{num_episodes} | Loss prédicteur: {np.mean(losses):.6f}")

		self.predictor.save_model("predictor.pt")

		if self.predictor_losses:
			show_loss(self.predictor_losses, window_size=10, path="predictor_training")

		return self.predictor_losses

	def train_agent_with_prediction(self, num_episodes=500, max_steps=4000, save_freq=100):
		for episode in range(1, num_episodes + 1):
			if not self.running:
				break

			state = self.reset()
			hidden = None
			episode_reward = 0
			episode_losses = []

			for step in range(1, max_steps + 1):
				action = self.agent.choose_action(state)

				next_state, reward, done = self.step(action)

				self.agent.replay_buffer.store_transition(state, action, reward, next_state, done)
				self.predictor.store_transition(state, action, next_state)

				loss = self.agent.learn()
				if loss is not None:
					episode_losses.append(loss)

				if step % self.delay == 0:
					state = next_state
					hidden = None
				else:
					state, hidden = self.predictor.predict_state(state, action, hidden)

				episode_reward += reward

				if self.visual:
					self.root.update_idletasks()
					self.root.update()

				if done:
					break

			if episode % 10 == 0:
				predictor_loss = self.predictor.learn(epoch=3)
				if predictor_loss:
					self.predictor_losses.extend(predictor_loss)

			self.episode_rewards.append(episode_reward)
			avg_loss = np.mean(episode_losses) if episode_losses else 0
			self.agent_losses.append(avg_loss)

			self.agent.epsilon_decay()

			print(f"Épisode {episode}/{num_episodes} | Récompense: {episode_reward:.2f} | "
				  f"Étapes: {step} | Epsilon: {self.agent.epsilon:.3f} | Loss: {avg_loss:.6f}")

		self.save_models("hybrid")

		if self.episode_rewards and self.agent_losses:
			show_result(self.episode_rewards, self.agent_losses, path="agent_training")

		return self.episode_rewards, self.agent_losses

	def train_agent(self, num_episodes=500, max_steps=4000, save_freq=100):
		for episode in range(1, num_episodes + 1):
			if not self.running:
				break

			state = self.reset()
			episode_reward = 0
			episode_losses = []

			for step in range(1, max_steps + 1):
				action = self.agent.choose_action(state)
				next_state, reward, done = self.step(action)

				self.agent.replay_buffer.store_transition(state, action, reward, next_state, done)

				loss = self.agent.learn()
				if loss is not None:
					episode_losses.append(loss)
				state = next_state
				episode_reward += reward
				if self.visual:
					self.root.update_idletasks()
					self.root.update()
				if done:
					break

			self.episode_rewards.append(episode_reward)
			avg_loss = np.mean(episode_losses) if episode_losses else 0
			self.agent_losses.append(avg_loss)
			self.agent.epsilon_decay()

			print(f"Épisode {episode}/{num_episodes} | Récompense: {episode_reward:.2f} | "
				  f"Étapes: {step} | Epsilon: {self.agent.epsilon:.3f} | Loss: {avg_loss:.6f}")
		self.agent.save_model("agent.pt")

		if self.episode_rewards and self.agent_losses:
			show_result(self.episode_rewards, self.agent_losses, path="agent_training")

		return self.episode_rewards, self.agent_losses

def play():
	height = 600
	width = 800
	delay = 60
	visual = True

	model_name = "agent.pt"

	trainer = PongTrainer(height, width, delay, visual)
	# trainer.load_models(model_name)
	trainer.agent.load_model(model_name)

	state = trainer.reset()
	hidden = None
	frame = 0
	while(1):
		action1 = trainer.agent.choose_action(state)
		action2 = trainer.get_key_action()
		next_state, _, terminal = trainer.step(action1, action2)

		# if frame % delay == 0:
		state = next_state
		# 	hidden = None
		# else:
		# 	state, hidden = trainer.predictor.predict_state(state, action1, hidden)

		if terminal:
			state = trainer.reset()
		trainer.root.update_idletasks()
		trainer.root.update()
		sleep(0.017)

def main():
	height = 600
	width = 800
	delay = 60
	visual = True

	trainer = PongTrainer(height, width, delay, visual)
	trainer.agent.load_model("agent.pt")
	trainer.train_agent(num_episodes=1000)
	trainer.running = True
	# trainer.train_predictor(num_episodes=500)
	# trainer.running = True
	# trainer.train_agent_with_prediction(num_episodes=1000)

if __name__ == "__main__":
	main()
	# play()
