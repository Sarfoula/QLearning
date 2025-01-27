import random
import numpy as np
import tkinter as tk
from ball import Ball
from paddle import Paddle

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

class Game:
	def __init__(self, root):
		self.root = root
		self.root.title("Jeu Pong")

		self.canvas = tk.Canvas(root, width=800, height=600, bg="black")
		self.canvas.pack()

		self.paddle_left = Paddle(self.canvas, 30, 250, color="red")
		self.paddle_right = Paddle(self.canvas, 750, 250, color="blue")
		self.ball = Ball(self.canvas, 390, 290, 20, "white", dx=random.randrange(-5, 6, 10), dy=0)
		self.ia = Agent()

		self.keys_pressed = {}
		self.root.bind("<KeyPress>", self.key_press)
		self.root.bind("<KeyRelease>", self.key_release)

		self.game_loop()

	def key_press(self, event):
		self.keys_pressed[event.keysym] = True

	def key_release(self, event):
		self.keys_pressed[event.keysym] = False

	def move_paddles(self, paddle, direction):
		paddle_coords = paddle.get_coords()
		if paddle_coords[1] - 10 >= 0 and direction == "Up":
			paddle.move(0, -10)
		if paddle_coords[3] + 10 <= 600 and direction == "Down":
			paddle.move(0, 10)

	def game_loop(self):
		self.Qlearning(self.paddle_left)
		self.Qlearning(self.paddle_right)
		victory = self.ball.check_collision_with_wall()
		if victory != 0:
			self.ball.reset()
		if self.ball.dx > 0:
			self.ball.check_collision_with_paddle(self.paddle_right)
		else:
			self.ball.check_collision_with_paddle(self.paddle_left)

		self.root.after(20, self.game_loop)

	def get_paddle_state(self, paddle):
		ball_coords = self.ball.get_coords()
		paddle_coords = paddle.get_coords()
		bx1, by1, bx2, by2 = ball_coords
		px1, py1, px2, py2 = paddle_coords
		if by1 > py1:
			return 0
		elif by2 < py2:
			return 1
		else:
			return 2

	def get_reward(self, state, action):
		if state == 0 and action == 1:
			return 1
		if state == 1 and action == 0:
			return 1
		if state == 2 and action == 2:
			return 1
		return -1

	def Qlearning(self, paddle):
		# Choose action
		current_state = self.get_paddle_state(paddle)
		action = self.ia.choose_action(current_state)

		# Execute action
		if action == 0:
			self.move_paddles(paddle, "Up")
		elif action == 1:
			self.move_paddles(paddle, "Down")

		# Get a reward
		reward = self.get_reward(current_state, action)

		# Update Q-table
		current_state = self.get_paddle_state(paddle)
		self.ia.update(current_state, action, reward, current_state)
		self.ia.exploration_prob /= 2

if __name__ == "__main__":
	root = tk.Tk()
	game = Game(root)
	root.mainloop()
