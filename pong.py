import tkinter as tk

class Ball:
	def __init__(self, visual, canvas, x, y, speed=10, dx=1, dy=0, radius=10, color="white"):
		self.start = [x, y]
		self.x = x
		self.y = y
		self.dx = dx
		self.dy = dy
		self.speed = speed
		self.vx = dx * speed
		self.vy = dy * speed
		self.radius = radius
		self.visual = visual
		if visual:
			self.canvas = canvas
			self.ball_id = self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=color)

	def move(self):
		self.vy = self.dy * self.speed
		self.vx = self.dx * self.speed
		self.x += self.vx
		self.y += self.vy
		if self.visual:
			self.canvas.move(self.ball_id, self.vx, self.vy)

	def get_center(self):
		return self.x, self.y

	def get_coords(self):
		if self.visual:
			return self.canvas.coords(self.ball_id)
		else:
			return self.x - self.radius, self.y - self.radius, self.x + self.radius, self.y + self.radius

	def reset(self):
		self.x = self.start[0]
		self.y = self.start[1]
		if self.visual:
			self.canvas.coords(self.ball_id, 380, 280, 400, 300)

class Paddle:
	def __init__(self, visual, canvas, x, y, color, width=10, height=100):
		self.x = x
		self.y = y
		self.start = [x, y]
		self.hit = 0
		self.width = width
		self.height = height
		self.visual = visual
		if visual:
			self.canvas = canvas
			self.paddle = self.canvas.create_rectangle(x - self.width/2, y - self.height/2, x + self.width/2, y + self.height/2, fill=color)

	def move_up(self):
		if self.y - self.height / 2 - 10 >= 0:
			movement = 10
		else:
			movement = self.y - self.height / 2
		self.y -= movement
		if self.visual:
			self.canvas.move(self.paddle, 0, -movement)

	def move_down(self):
		if self.y + self.height / 2 + 10 <= 600:
			movement = 10
		else:
			movement = 600 - (self.y + self.height / 2)
		self.y += movement
		if self.visual:
			self.canvas.move(self.paddle, 0, movement)

	def move(self, dx, dy):
		self.y += dy
		self.canvas.move(self.paddle, dx, dy)

	def reset(self):
		self.hit = 0
		self.y = self.start[1]
		if self.visual:
			self.canvas.coords(self.paddle, self.x - self.width/2, self.y - self.height/2, self.x + self.width/2, self.y + self.height/2)

	def get_coords(self):
		if self.visual:
			return self.canvas.coords(self.paddle)
		else:
			return self.x - self.width / 2, self.y - self.height / 2, self.x + self.width / 2, self.y + self.height

	def get_center(self):
		return self.x, self.y

class Game:
	def __init__(self, height, width, visual=True):
		self.winner = 0

		self.width = width
		self.height = height
		self.ballspeed = 10
		self.canvas = None
		self.keys_pressed = {}
		if visual:
			self.root = tk.Tk()
			self.root.title("Jeu Pong")
			self.canvas = tk.Canvas(self.root, width=width, height=height, bg="black")
			self.canvas.pack()
			self.root.bind("<KeyPress>", self.key_press)
			self.root.bind("<KeyRelease>", self.key_release)

		self.ball = Ball(visual, self.canvas, x=width/2, y=height/2, dx=1, dy=0, speed=self.ballspeed, radius=10, color="white")
		self.paddle_left = Paddle(visual, self.canvas, 50, height/2, width=10, height=100, color="red")
		self.paddle_right = Paddle(visual, self.canvas, width - 50, height/2, width=10, height=100, color="blue")

	def key_press(self, event):
		self.keys_pressed[event.keysym] = True

	def key_release(self, event):
		self.keys_pressed[event.keysym] = False

	def get_key_action(self):
		if self.keys_pressed.get("w"):
			return 0
		elif self.keys_pressed.get("s"):
			return 1
		return 2

	def move_paddle(self, paddle, action):
		if action == 0:
			paddle.move_up()
		elif action == 1:
			paddle.move_down()

	def opponent(self, paddle):
		if self.ball.y < paddle.y:
			return 0
		elif self.ball.y > paddle.y:
			return 1
		else:
			return 2

	def reset(self):
		self.paddle_right.reset()
		self.paddle_left.reset()
		self.ball.reset()
		self.ball_hit = False
		self.winner = 0

	def check_collision_paddle(self, ball, paddle):
		_, by = ball.get_center()
		px, py = paddle.get_center()
		px1, py1, px2, py2 = paddle.get_coords()
		bx1, by1, bx2, by2 = ball.get_coords()

		if bx2 >= px1 and bx1 <= px2 and by2 >= py1 and by1 <= py2:
			angle = (by - py) / (py2 - py1)
			if px < self.width / 2:
				ball.dx = 1 - abs(angle)
			else:
				ball.dx = -(1 - abs(angle))
			ball.dy = angle
			paddle.hit += 1
			return True
		return False

	def check_collision_wall(self, ball):
		"""return zero if no victory, 1 for Left and 2 for Right"""
		bx, by = ball.get_center()

		if by - ball.radius <= 0:
			ball.dy = abs(ball.dy)
		if by + ball.radius >= self.height:
			ball.dy = -abs(ball.dy)
		if bx + ball.radius >= self.width:
			ball.dx = -abs(ball.dx)
			self.winner = 1
			print("GAUCHE won")
		if bx - ball.radius <= 0:
			ball.dx = abs(ball.dx)
			self.winner = 2
			print("DROITE won")

	def get_state(self):
		"""return paddle_y, opponent_y, ball_x, ball_y, ball_vx, ball_vy"""
		return (self.paddle_left.y/self.height, self.paddle_right.y/self.height,
				self.ball.x/self.width, self.ball.y/self.height,
				self.ball.vx/self.ballspeed, self.ball.vy/self.ballspeed)

	def step(self, left_action, right_action):
		self.ball.move()
		self.move_paddle(self.paddle_left, left_action)
		self.move_paddle(self.paddle_right, right_action)

		self.check_collision_wall(self.ball)
		self.check_collision_paddle(self.ball, self.paddle_left)
		self.ball_hit = self.check_collision_paddle(self.ball, self.paddle_right)

		if self.winner != 0:
			return True
		return False

	def game_loop(self):
		left_action = self.simpleOpponent(self.paddle_left)
		right_action = self.get_key_action()

		self.step(left_action, right_action)

		if self.winner != 0:
			self.reset()
		self.root.after(17, self.game_loop)

if __name__ == "__main__":
	game = Game(600, 800, visual=True)
	game.game_loop()
	game.root.mainloop()
