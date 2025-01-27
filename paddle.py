class Paddle:
	def __init__(self, canvas, x, y, color):
		self.canvas = canvas
		self.width = 10
		self.height = 100
		self.paddle = self.canvas.create_rectangle(x, y, x + self.width, y + self.height, fill=color)

	def move(self, dx, dy):
		self.canvas.move(self.paddle, dx, dy)

	def get_coords(self):
		return self.canvas.coords(self.paddle)

	def get_center(self):
		return self.get_coords()[1] + self.height / 2
