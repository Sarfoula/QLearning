class Paddle:
	def __init__(self, canvas, x, y, color):
		self.canvas = canvas
		self.width = 10
		self.height = 100
		self.x = x
		self.y = y
		self.starty = y
		self.startx = x
		self.paddle = self.canvas.create_rectangle(x - self.width/2, y - self.height/2, x + self.width/2, y + self.height/2, fill=color)

	def move_up(self):
		if self.y - self.height / 2 - 20 >= 0:
			movement = 20
		else:
			movement = self.y - self.height / 2
		self.y -= movement
		self.canvas.move(self.paddle, 0, -movement)

	def move_down(self):
		if self.y + self.height / 2 + 20 <= 600:
			movement = 20
		else:
			movement = 600 - (self.y + self.height / 2)
		self.y += movement
		self.canvas.move(self.paddle, 0, movement)

	def move(self, dx, dy):
		self.x += dx
		self.y += dy
		self.canvas.move(self.paddle, dx, dy)

	def reset(self):
		self.x = self.startx
		self.y = self.starty
		self.canvas.coords(self.paddle, self.x - self.width/2, self.y - self.height/2, self.x + self.width/2, self.y + self.height/2)

	def get_coords(self):
		return self.canvas.coords(self.paddle)

	def get_center(self):
		return self.x, self.y
