import random

class Ball:
	def __init__(self, canvas, x, y, radius, color, dx, dy, speed):
		self.canvas = canvas
		self.radius = radius
		self.speed = speed
		self.dx = dx
		self.dy = dy
		self.vx = dx * speed
		self.vy = dy * speed
		self.ball_id = canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=color)

	def move(self):
		self.vx = self.speed * self.dx
		self.vy = self.speed * self.dy
		self.canvas.move(self.ball_id, self.vx, self.vy)

	def get_center(self):
		coords = self.get_coords()
		return coords[0] + self.radius, coords[1] + self.radius

	def get_coords(self):
		return self.canvas.coords(self.ball_id)

	def reset(self):
		self.canvas.coords(self.ball_id, 380, 280, 400, 300)
