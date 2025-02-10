import random

class Ball:
	def __init__(self, canvas, x, y, dx, dy, radius, color, speed):
		self.x = x
		self.y = y
		self.dx = dx
		self.dy = dy
		self.speed = speed
		self.vx = self.dx * self.speed
		self.vy = self.dy * self.speed
		self.canvas = canvas
		self.radius = radius
		self.ball_id = canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=color)

	def move(self):
		self.vx = self.dx * self.speed
		self.vy = self.dy * self.speed
		self.x += self.vx
		self.y += self.vy
		self.canvas.move(self.ball_id, self.vx, self.vy)

	def bounce_x(self):
		self.vx = -self.vx

	def bounce_y(self):
		self.vy = -self.vy

	def get_center(self):
		return self.x, self.y

	def get_coords(self):
		return self.canvas.coords(self.ball_id)

	def reset(self):
		self.x = 400
		self.y = 300
		self.dx = -0.9
		self.dy = 0.1
		self.vx = self.dx * self.speed
		self.vy = self.dy * self.speed
		self.canvas.coords(self.ball_id, 380, 280, 400, 300)
