import tkinter as tk
import numpy as np
from ball import Ball
from paddle import Paddle
from DeepQlearning import Agent
from utils import show_result

class Game:
	def __init__(self, height, width):
		self.winner = 0
		self.hit = 0

		self.root = tk.Tk()
		self.width = width
		self.height = height
		self.agent = Agent(gamma=0.99, batch_size=64, n_actions=3, input_dims=6, lr=0.00025, reward_factor=0.01, tau=1000)
		self.root.title("Jeu Pong")

		self.canvas = tk.Canvas(self.root, width=width, height=height, bg="black")
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

	def move_paddle_key(self, paddle):
		if self.keys_pressed.get("w"):
			return 0
		elif self.keys_pressed.get("s"):
			return 1
		return 2

	def move_paddle_action(self, paddle, action):
		if action == 0:
			paddle.move_up()
		elif action == 1:
			paddle.move_down()

	def simpleOpponent(self, paddle):
		ball_c = self.ball.get_center()
		paddle_c = paddle.get_center()
		move = False
		if paddle_c[0] < self.width / 2 and self.ball.dx < 0:
			move = True
		elif paddle_c[0] > self.width / 2 and self.ball.dx > 0:
			move = True
		if ball_c[1] > paddle_c[1] + 50 and move:
			return 1
		elif ball_c[1] < paddle_c[1] - 50 and move:
			return 0
		return np.random.choice([0, 1, 2])

	def reset(self, paddle):
		self.paddle_right.reset()
		self.paddle_left.reset()
		self.ball.reset()
		self.winner = 0
		self.hit = 0

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
			if px1 < self.width / 2:
				ball.dx = 1 - abs(angle)
			else:
				ball.dx = -(1 - abs(angle))
			ball.dy = angle
			self.hit += 1
			paddle.ball_hit = True

	def check_collision_with_wall(self, ball):
		"""return zero if no victory, 1 for Right and 2 for Left"""
		bx1, by1, bx2, by2 = ball.get_coords()

		if by1 <= 0:
			ball.dy = abs(ball.dy)
		if by2 >= self.canvas.winfo_height():
			ball.dy = -abs(ball.dy)
		if bx1 <= 0:
			ball.dx = -ball.dx
			self.winner = 1
			print("Right Paddle won")
		if bx2 >= self.width:
			ball.dx = -ball.dx
			self.winner = 2
			print("Left Paddle won")

	def get_state(self, paddle, opponent):
		return np.array([abs(paddle.x - self.ball.x), self.ball.y, self.ball.dx, self.ball.dy, paddle.y, opponent.y])

	def step(self, left, right):
		self.ball.move()
		self.move_paddle_action(self.paddle_left, left)
		self.move_paddle_action(self.paddle_right, right)

		self.check_collision_with_wall(self.ball)
		self.check_collision_with_paddle(self.ball, self.paddle_left)
		self.check_collision_with_paddle(self.ball, self.paddle_right)

		ball_hit = self.paddle_left.ball_hit or self.paddle_right.ball_hit
		if ball_hit:
			self.paddle_left.ball_hit = False
			self.paddle_right.ball_hit = False

		if self.winner != 0:
			return True, ball_hit
		return False, ball_hit

	def get_reward(self, paddle, opponent):
		ball_center = self.ball.get_center()
		paddle_center = paddle.get_center()
		bx1, by1, bx2, by2 = self.ball.get_coords()
		px1, py1, px2, py2 = paddle.get_coords()
		reward = 0

		if bx2 >= px1 and bx1 <= px2 and by2 >= py1 and by1 <= py2:
			reward = 1
		px1, py1, px2, py2 = opponent.get_coords()
		if bx2 >= px1 and bx1 <= px2 and by2 >= py1 and by1 <= py2 and self.hit > 0:
			reward = -0.1
		distance = round(abs(ball_center[1] - paddle_center[1]) / 60)
		if paddle.x < self.width / 2:
			if self.winner == 1:
				reward = -distance
			elif self.winner == 2 and self.hit > 0:
				reward = distance
		else:
			if self.winner == 1 and self.hit > 0:
				reward = distance
			elif self.winner == 2:
				reward = -distance
		return reward

	def game_train(self, num_games=100):
		scores = []
		paddle = self.paddle_left
		opponent = self.paddle_right

		for i in range(num_games):
			done = False
			self.reset(paddle)
			state = self.get_state(paddle, opponent)
			score = 0
			print('episode', i)
			while not done:
				action_left = self.agent.choose_action(state)
				action_right = self.simpleOpponent(opponent)

				done, ball_hit = self.step(action_left, action_right)
				new_state = self.get_state(paddle, opponent)
				reward_left = self.get_reward(paddle, opponent)

				score += reward_left

				self.agent.replayBuffer.store_transition(state, action_left, reward_left, new_state, done, ball_hit)
				self.agent.learn()
				state = new_state
			self.agent.epsilon = self.agent.epsilon - 0.001 if self.agent.epsilon > 0.01 else 0.01
			scores.append(score)
			avg_score = np.mean(scores)
			print('score %.2f' % score, 'average %.2f' % avg_score, 'hit', self.hit)
		show_result(scores, num_games)

	def game_loop(self):
		# get an action for AI
		left_action = self.agent.choose_action(self.get_state(self.paddle_left, self.paddle_right))
		# left_action = self.simpleOpponent(self.paddle_left)
		right_action = self.move_paddle_key(self.paddle_right)

		self.step(left_action, right_action)

		# Check for winner and do it again
		reward = self.get_reward(self.paddle_left, self.paddle_right)
		if not reward == 0:
			print(reward)
		if self.winner != 0:
			self.reset(self.paddle_left)
		self.root.after(50, self.game_loop)

if __name__ == "__main__":
	game = Game(600, 800)
	game.game_train(500)
	game.agent.epsilon = 0
	game.game_loop()
	game.root.mainloop()
