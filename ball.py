import random

class Ball:
	def __init__(self, canvas, x, y, diameter, color, dx, dy):
		self.canvas = canvas
		self.diameter = diameter
		self.radius = diameter / 2
		self.dx = dx
		self.dy = dy
		self.ball_id = canvas.create_oval(x, y, x + diameter, y + diameter, fill=color)

	def check_collision_with_wall(self):
		"""return zero if no victory, 1 for RED and 2 for BLUE"""
		x1, y1, x2, y2 = self.canvas.coords(self.ball_id)
		next_x1 = x1 + self.dx
		next_y1 = y1 + self.dy
		next_x2 = x2 + self.dx
		next_y2 = y2 + self.dy

		if next_y1 <= 0 or next_y2 >= self.canvas.winfo_height():
			self.dy = -self.dy
		self.canvas.move(self.ball_id, self.dx, self.dy)
		if next_x1 <= 0:
			print("BLUE WIN")
			return 2
		if next_x2 >= 800:
			print("RED WIN")
			return 1
		return 0

	def get_center(self):
		return self.canvas.coords(self.ball_id)[1] + self.radius

	def get_coords(self):
		return self.canvas.coords(self.ball_id)

	def check_collision_with_paddle(self, paddle):
		ball_coords = self.canvas.coords(self.ball_id)
		next_x1 = ball_coords[0] + self.dx
		next_y1 = ball_coords[1] + self.dy
		next_x2 = ball_coords[2] + self.dx
		next_y2 = ball_coords[3] + self.dy

		paddle_coords = paddle.get_coords()
		px1, py1, px2, py2 = paddle_coords

		paddle_center = paddle.get_center()
		ball_center = self.get_center()
		if next_x2 >= px1 and next_x1 <= px2 and next_y2 >= py1 and next_y1 <= py2:
			if self.dx > 0:
				self.canvas.move(self.ball_id, px1 - next_x2, 0)
			elif self.dx < 0:
				self.canvas.move(self.ball_id, px2 - next_x1, 0)
			self.dx = -self.dx
			self.dy = (ball_center - paddle_center) / 2

	def reset(self):
		self.dy = random.randrange(-3, 3, 1)
		self.canvas.coords(self.ball_id, 380, 280, 400, 300)
