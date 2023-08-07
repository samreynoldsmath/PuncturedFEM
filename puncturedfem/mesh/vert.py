class vert:
	"""Stores physical location of a mesh vertex"""

	id: any
	x: float
	y: float

	def __init__(self, x: float, y: float, id=None) -> None:
		self.set_coord(x, y)
		self.set_id(id)

	def set_id(self, id: any) -> None:
		self.id = id

	def set_coord(self, x: float, y: float) -> None:
		if isinstance(x, int):
			x = float(x)
		if isinstance(y, int):
			y = float(y)
		if not isinstance(x, float) or not isinstance(y, float):
			raise TypeError("Coordinates x and y must be floats")
		self.x = x
		self.y = y

	def __eq__(self, other: object) -> bool:
		return self.x == other.x and self.y == other.y