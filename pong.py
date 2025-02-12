import tkinter as tk
import numpy as np
from ball import Ball
from paddle import Paddle
from DeepQlearning import Agent

import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import torch

class Game:
	def __init__(self, root, height, width):
		self.winner = 0
		self.collision = 0
		self.breaking = 0

		self.root = root
		self.agent = Agent(gamma=0.99, epsilon=1.0, eps_decay=0.005, batch_size=128, n_actions=3, input_dims=5, lr=0.001)
		self.opponent = self.agent
		self.root.title("Jeu Pong")

		self.canvas = tk.Canvas(root, width=width, height=height, bg="black")
		self.canvas.pack()

		self.ball = Ball(self.canvas, x=width/2, y=height/2, dx=-0.9, dy=0.1, radius=10, speed=30, color="white")
		self.paddle_left = Paddle(self.canvas, 50, height/2, color="red")
		self.paddle_right = Paddle(self.canvas, width - 50, height/2, color="blue")

		self.keys_pressed = {}
		self.root.bind("<KeyPress>", self.key_press)
		self.root.bind("<KeyRelease>", self.key_release)

	def key_press(self, event):
		self.keys_pressed[event.keysym] = True

	def key_release(self, event):
		self.keys_pressed[event.keysym] = False

	def move_paddle(self):
		if self.keys_pressed.get("w"):
			self.paddle_left.move_up()
		elif self.keys_pressed.get("s"):
			self.paddle_left.move_down()

	def simpleOpponent(self, paddle):
		ball_c = self.ball.get_center()
		paddle_c = paddle.get_center()
		if ball_c[1] > paddle_c[1] + 20:
			paddle.move_down()
		elif ball_c[1] < paddle_c[1] - 20:
			paddle.move_up()

	def get_state(self):
		return np.array([self.ball.x, self.ball.y, self.ball.dx, self.ball.dy, self.paddle_right.y, self.paddle_left.y])

	def reset(self):
		self.paddle_right.reset()
		self.paddle_left.reset()
		self.ball.reset()
		self.winner = 0
		return self.get_state()

	def check_collision_with_paddle(self, ball, paddle):
		paddle_center = paddle.get_center()
		ball_center = ball.get_center()
		ball_coords = ball.get_coords()
		paddle_coords = paddle.get_coords()
		bx1, by1, bx2, by2 = ball_coords
		px1, py1, px2, py2 = paddle_coords

		if bx2 >= px1 and bx1 <= px2 and by2 >= py1 and by1 <= py2:
			angle = (ball_center[1] - paddle_center[1]) / 50
			if abs(angle) > 0.6:
				angle = 0.6 if angle > 0 else -0.6
			if px1 == 45:
				ball.dx = 1 - abs(angle)
			else:
				ball.dx = -(1 - abs(angle))
			ball.dy = angle
			return 1
		return 0

	def check_collision_with_wall(self, ball):
		"""return zero if no victory, 1 for IA and 2 for BOT"""
		bx1, by1, bx2, by2 = ball.get_coords()

		if by1 <= 0 or by2 >= self.canvas.winfo_height():
			ball.dy = -ball.dy
		if bx1 <= 0:
			ball.dx = -ball.dx
			print("RIGHT PADDLE WONnnnnNN")
			self.winner = 2
		if bx2 >= 800:
			ball.dx = -ball.dx
			print("LEFT PADDLE WON")
			self.winner = 1

	def step(self, action1, action2):
		# Moving
		self.ball.move()

		if action1 == 0:
			self.paddle_left.move_up()
		elif action1 == 1:
			self.paddle_left.move_down()
		if action2 == 0:
			self.paddle_right.move_up()
		elif action2 == 1:
			self.paddle_right.move_down()

		# Colision
		self.check_collision_with_wall(self.ball)
		self.collision += self.check_collision_with_paddle(self.ball, self.paddle_left)
		self.collision += self.check_collision_with_paddle(self.ball, self.paddle_right)

	def get_reward(self, action):
		ball = self.ball.get_center()
		paddle = self.paddle_right.get_center()
		bx1, by1, bx2, by2 = self.ball.get_coords()
		px1, py1, px2, py2 = self.paddle_right.get_coords()
		reward = 0

		if bx2 >= px1 and bx1 <= px2 and by2 >= py1 and by1 <= py2:
			reward = 1
		px1, py1, px2, py2 = self.paddle_left.get_coords()
		if bx2 >= px1 and bx1 <= px2 and by2 >= py1 and by1 <= py2:
			reward = -1
		if self.winner == 2:
			distance = (int)(math.dist(paddle, ball) / 100)
			reward = -abs(distance)
		if self.winner == 1:
			reward = 10
		return reward

	def game_train(self, num_games=100):

		scores = []

		for i in range(num_games):

			done = False
			state = self.reset()
			score = 0

			while not done:
				action = self.agent.choose_action(state)
				new_state, reward, done = self.step(action)

				score += reward

				self.agent.store_transition(state, action, reward, new_state, done)
				self.agent.learn()
				state = new_state

			scores.append(score)
			avg_score = np.mean(scores)
			print('episode', i, 'score %.2f' % score, 'average %.2f' % avg_score, 'epsilon %.2f' % self.agent.epsilon, 'hit', self.collision)

			self.agent.epsilon = max(self.agent.eps_end, self.agent.eps_start - (self.agent.eps_start - self.agent.eps_end) * (i / num_games))

		# torch.save(self.agent.Q_eval, "Mybrain.pth")
		# Tracé des courbes
		# smooth = np.convolve(scores, np.ones(10) / 10, mode='valid')
		# plt.xlabel("Épisode")
		# plt.ylabel("Récompenses")
		# plt.plot(range(1, num_games + 1), scores, label="Scores", alpha=0.3, color="blue")
		# plt.plot(range(10, num_games + 1), smooth, label="Average", color="red")
		# plt.axhline(0, color="black", linestyle="dotted")
		# plt.legend()
		# plt.show()

	def get_state_double(self, paddle):
		if paddle == 1:
			return np.array([self.ball.x, self.ball.y, self.ball.dx, self.ball.dy, self.paddle_left.y])
		else:
			return np.array([800 - self.ball.x, 600 - self.ball.y, -self.ball.dx, -self.ball.dy, self.paddle_right.y])

	def get_reward_double(self, paddle):
		ball = self.ball.get_center()
		bx1, by1, bx2, by2 = self.ball.get_coords()
		reward = 0

		if paddle == 1:
			paddle = self.paddle_left.get_center()
			px1, py1, px2, py2 = self.paddle_left.get_coords()

			if bx2 >= px1 and bx1 <= px2 and by2 >= py1 and by1 <= py2:
				reward = 1
			# px1, py1, px2, py2 = self.paddle_right.get_coords()
			# if bx2 >= px1 and bx1 <= px2 and by2 >= py1 and by1 <= py2:
			# 	reward = -1
			if self.winner == 1:
				reward = 10
			if self.winner == 2:
				distance = (int)(math.dist(paddle, ball) / 100)
				reward = -abs(distance)
		else:
			paddle = self.paddle_right.get_center()
			px1, py1, px2, py2 = self.paddle_right.get_coords()

			if bx2 >= px1 and bx1 <= px2 and by2 >= py1 and by1 <= py2:
				reward = 1
			# px1, py1, px2, py2 = self.paddle_left.get_coords()
			# if bx2 >= px1 and bx1 <= px2 and by2 >= py1 and by1 <= py2:
			# 	reward = -1
			if self.winner == 2:
				reward = 10
			if self.winner == 1:
				distance = (int)(math.dist(paddle, ball) / 100)
				reward = -abs(distance)
			return reward
		return reward

	def game_double_agent(self, num_games=100):
		for i in range(num_games):
			print("episode ", i, " collisions ", self.collision)
			self.collision = 0
			done = False
			self.reset()
			state1 = self.get_state_double(1)
			state2 = self.get_state_double(2)

			while not done:
				action1 = self.agent.choose_action(state1)
				action2 = self.agent.choose_action(state2)

				self.step(action1, action2)

				new_state1 = self.get_state_double(1)
				new_state2 = self.get_state_double(2)
				reward1 = self.get_reward_double(1)
				reward2 = self.get_reward_double(2)
				if self.winner != 0:
					done = True

				self.agent.store_transition(state1, action1, reward1, new_state1, done)
				self.agent.store_transition(state2, action2, reward2, new_state2, done)

				self.agent.learn

				state1 = new_state1
				state2 = new_state2

			self.agent.epsilon = max(self.agent.eps_end, self.agent.eps_start - (self.agent.eps_start - self.agent.eps_end) * (i / num_games))

	def game_loop(self):
		# get an action for AI
		state = self.get_state_double(2)
		action = self.agent.choose_action(state)

		# Moving
		if action == 0:
			self.paddle_right.move_up()
		elif action == 1:
			self.paddle_right.move_down()
		self.move_paddle()
		self.ball.move()

		# Colision
		self.check_collision_with_wall(self.ball)
		self.check_collision_with_paddle(self.ball, self.paddle_left)
		self.check_collision_with_paddle(self.ball, self.paddle_right)

		# Check for winner and do it again
		if self.winner != 0:
			self.reset()
		self.root.after(50, self.game_loop)

	def game_default(self):
		action = 2
		if self.keys_pressed.get("w"):
			action = 0
		if self.keys_pressed.get("s"):
			action = 1
		state, reward, done = self.step(action)

		print(reward)
		# Check for winner and do it again
		if self.winner != 0:
			self.reset()
		self.root.after(100, self.game_default)

if __name__ == "__main__":
	root = tk.Tk()
	game = Game(root, 600, 800)
	# game.game_train(1000)
	game.game_double_agent(5000)
	game.agent.epsilon = 0
	game.game_loop()
	# game.game_default()
	root.mainloop()
