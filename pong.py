import tkinter as tk
from ball import Ball
from paddle import Paddle
from math import *
from agent import DQNAgent
import numpy as np

class Game:
	def __init__(self, root, height, width, game_speed):
		self.root = root
		self.game_speed = game_speed
		self.agent = DQNAgent(input_dim=5, output_dim=5)
		self.root.title("Jeu Pong")

		self.canvas = tk.Canvas(root, width=width, height=height, bg="black")
		self.canvas.pack()

		self.paddle_left = Paddle(self.canvas, 50, height/2, color="red")
		self.paddle_right = Paddle(self.canvas, width - 50, height/2, color="blue")
		self.ball = Ball(self.canvas, width/2, height/2, 10, "white",  dx=0.95, dy=0.05, speed=5)

		self.game_loop()

	def key_press(self, event):
		self.keys_pressed[event.keysym] = True

	def key_release(self, event):
		self.keys_pressed[event.keysym] = False

	def check_collision_with_paddle(self, ball, paddle):
		paddle_center = paddle.get_center()
		ball_center = self.ball.get_center()
		ball_coords = self.ball.get_coords()
		paddle_coords = paddle.get_coords()
		bx1, by1, bx2, by2 = ball_coords
		px1, py1, px2, py2 = paddle_coords

		if bx2 >= px1 and bx1 <= px2 and by2 >= py1 and by1 <= py2:
			angle = (ball_center[1] - paddle_center[1]) / 50
			if ball.dx < 0:
				ball.dx = 1 - abs(angle)
			else:
				ball.dx = -(1 - abs(angle))
			ball.dy = angle

	def check_collision_with_wall(self, ball):
		"""return zero if no victory, 1 for RED and 2 for BLUE"""
		ball_coords = ball.get_coords()
		bx1, by1, bx2, by2 = ball_coords

		if by1 <= 0 or by2 >= self.canvas.winfo_height():
			ball.dy = -ball.dy
		if bx1 <= 0:
			ball.dx = -ball.dx
			print("BLUE WIN")
			ball.reset()
		if bx2 >= 800:
			ball.dx = -ball.dx
			print("RED WIN")
			ball.reset()

	def move_paddles(self, paddle, direction):
		paddle_coords = paddle.get_coords()
		if paddle_coords[1] - 10 >= 0 and direction == "Up":
			paddle.move(0, -10)
		elif paddle_coords[3] + 10 <= 600 and direction == "Down":
			paddle.move(0, 10)

	def simpleOpponent(self, paddle):
		ball_c = self.ball.get_center()
		paddle_c = paddle.get_center()
		if ball_c[1] > paddle_c[1]:
			self.move_paddles(paddle, "Down")
		elif ball_c[1] < paddle_c[1]:
			self.move_paddles(paddle, "Up")

	def get_reward(self):
		ball_coords = self.ball.get_coords()
		paddle_coords = self.paddle_right.get_coords()
		bx1, by1, bx2, by2 = ball_coords
		px1, py1, px2, py2 = paddle_coords

		if bx2 >= px1 and bx1 <= px2 and by2 >= py1 and by1 <= py2:
			reward += 1
		if bx2 >= px2:
			reward -= 1
		return reward

	def DQN(self, state):
		# Exploitation vs exploration
		action = self.agent.choose_action(state)

		if action == 0:
			self.move_paddles(self.paddle_right, "Up")
		elif action == 1:
			self.move_paddles(self.paddle_right, "Down")

		# Calculate reward
		reward = self.agent.get_reward(state)
		done = self.is_done(state)

		next_state = np.array([[self.ball.get_center[0], self.ball.get_center[1], self.ball.vx, self.ball.vy, self.paddle_right.get_center()[1]]])
		self.agent.update(state, action, reward, next_state, done)

		self.agent.decrease_exploration_prob()

	def game_loop(self):
		self.simpleOpponent(self.paddle_left)
		state = np.array([[self.ball.get_center[0], self.ball.get_center[1], self.ball.vx, self.ball.vy, self.paddle_right.get_center()[1]]])
		self.agent.DQN(state)
		self.ball.move()
		self.check_collision_with_paddle(self.ball, self.paddle_left)
		self.check_collision_with_paddle(self.ball, self.paddle_right)
		self.check_collision_with_wall(self.ball)
		self.root.after((int)(20 / self.game_speed), self.game_loop)

if __name__ == "__main__":
	root = tk.Tk()
	game = Game(root, 600, 800, 1)
	root.mainloop()
