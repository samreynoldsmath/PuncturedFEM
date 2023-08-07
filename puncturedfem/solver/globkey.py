class global_key:
	"""Global key for local functions"""

	fun_type: str
	vert_idx: int
	edge_idx: int
	# cell_idx: int
	edge_space_idx: int
	bubb_space_idx: int
	glob_idx: int
	is_on_boundary: bool

	def __init__(self,
	      fun_type: str,
		  edge_idx: int=-1,
		  vert_idx: int=-1,
		#   cell_idx: int=-1,
		  bubb_space_idx: int=-1,
		  edge_space_idx: int=-1) -> None:
		self.set_fun_type(fun_type)
		self.set_vert_idx(vert_idx)
		self.set_edge_idx(edge_idx)
		# self.set_cell_idx(cell_idx)
		self.set_edge_space_idx(edge_space_idx)
		self.set_bubb_space_idx(bubb_space_idx)

	def set_fun_type(self, fun_type: str) -> None:
		if fun_type not in ['vert', 'edge', 'bubb']:
			raise ValueError('fun_type must be vert, edge, or bubb')
		self.fun_type = fun_type

	def set_vert_idx(self, vert_idx: int) -> None:
		if not isinstance(vert_idx, int):
			raise TypeError('vert_idx must be an integer')
		if not self.fun_type == 'vert':
			self.vert_idx = -1
		else:
			self.vert_idx = vert_idx

	def set_edge_idx(self, edge_idx: int) -> None:
		if not isinstance(edge_idx, int):
			raise TypeError('edge_idx must be an integer')
		if not self.fun_type == 'edge':
			self.edge_idx = -1
		else:
			self.edge_idx = edge_idx

	# def set_cell_idx(self, cell_idx: int) -> None:
	# 	if not isinstance(cell_idx, int):
	# 		raise TypeError('cell_idx must be an integer')
	# 	if cell_idx < 0:
	# 		raise ValueError('cell_idx must be nonnegative')
	# 	self.cell_idx = cell_idx

	def set_edge_space_idx(self, edge_space_idx: int) -> None:
		if not isinstance(edge_space_idx, int):
			raise TypeError('edge_space_idx must be an integer')
		if not self.fun_type == 'edge':
			self.edge_space_idx = -1
		else:
			self.edge_space_idx = edge_space_idx

	def set_bubb_space_idx(self, bubb_space_idx: int) -> None:
		if not isinstance(bubb_space_idx, int):
			raise TypeError('bubb_space_idx must be an integer')
		if not self.fun_type == 'bubb':
			self.bubb_space_idx = -1
		else:
			self.bubb_space_idx = bubb_space_idx

	def set_glob_idx(self, glob_idx: int) -> None:
		if not isinstance(glob_idx, int):
			raise TypeError('glob_idx must be an integer')
		if glob_idx < 0:
			raise ValueError('glob_idx must be nonnegative')
		self.glob_idx = glob_idx

	# def find_global_index(self):
	# 	"""Find the global index of the local function"""
	# 	if self.fun_type == 'vert':
	# 		self.glob_idx = self.vert_idx
	# 	elif self.fun_type == 'edge':
