import numpy as np
import random
import math
import time
import cProfile
import torch.multiprocessing as mp
import psutil


import os
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.optim import Adam

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

class Connect4:
	
	BOARD_WIDTH = 7
	BOARD_HEIGHT = 6
	IN_A_ROW = 4

	def __init__(self, board:np.array, player: int, num_moves, i, j):
		# if board.shape != (Connect4.BOARD_HEIGHT, Connect4.BOARD_WIDTH):
		# 	raise Exception("Invalid board dimensions!")
		self.board = board # 6 x 7 numpy matrix
		self.player = player # -1 or 1
		self.children = False
		
		self.num_moves = num_moves
		# self.check_terminality(i,j)
		self.i = i
		self.j = j
		self.is_terminal_cache = -1
		self.reward = 0

	def is_terminal(self):
		if self.is_terminal_cache != -1:
			return self.is_terminal_cache

		if self.num_moves == 0:
			self.is_terminal_cache = False
			self.reward = 0
			return False

		self.is_terminal_cache = self.num_moves == Connect4.BOARD_HEIGHT * Connect4.BOARD_WIDTH
		self.reward = 0
		
		for (idir, jdir) in [(1,0), (0,1), (1,1), (1,-1)]:
			self.check_4_in_row(self.i,self.j, idir,jdir)
		
		return self.is_terminal_cache
	
	def get_reward(self):
		if self.is_terminal_cache == -1:
			self.is_terminal()
		return self.reward

	def check_4_in_row(self, i,j, i_dir, j_dir):
		color = self.board[i][j]

		TOT = 1
		i1, j1 = i + i_dir, j + j_dir
		while i1 < Connect4.BOARD_HEIGHT and j1 < Connect4.BOARD_WIDTH and i1 >= 0 and j1 >= 0 and self.board[i1][j1] == color:
			TOT += 1
			i1, j1 = i1 + i_dir, j1 + j_dir

		i1, j1 = i - i_dir, j - j_dir
		while i1 < Connect4.BOARD_HEIGHT and j1 < Connect4.BOARD_WIDTH and i1 >= 0 and j1 >= 0 and self.board[i1][j1] == color:
			TOT += 1
			i1, j1 = i1 - i_dir, j1 - j_dir
		
		if TOT >= 4:
			self.is_terminal_cache = True
			self.reward = color
			return color

		return 0

	def __str__(self):
		output_str = "Connect4 Board!\n---------------\n"
		for i in reversed(range(Connect4.BOARD_HEIGHT)):
			output_str += '|'
			for j in range(Connect4.BOARD_WIDTH):
				output_str += 'L' if self.board[i][j] == 1 else ( 'R' if self.board[i][j] == -1 else ' ')
				output_str += '|'
			output_str += "\n---------------\n"

		return output_str

	def get_children(self):
		if self.children:
			return self.children
		
		self.children = []
		for i in range(Connect4.BOARD_WIDTH):
			if self.board[Connect4.BOARD_HEIGHT-1][i]:
				self.children.append(None)
			else:
				
				j = 0
				while self.board[j][i]:
					j = j+1
				
				new_board = np.copy(self.board)
				new_board[j][i] = self.player

				child = Connect4(new_board,-self.player,self.num_moves+1,j,i)
				self.children.append(child)
		
		return self.children

class MCTSNode:

	EXPLORATION_CONSTANT = 3
	
	DEFAULT_BOARD = Connect4( np.zeros((6,7)), 1, 0, 1, 1 )

	def __init__(self, parent, game: Connect4):
		self.game = game
		self.children = False

		self.parent = parent
		
		self.Vsum = 0
		self.N = 0
		self.P = [] # prior move probabilities

		self.disable_backprop = False

	def is_terminal(self):
		return self.game.is_terminal()

	def UCB(self, child_index):
		# if self.is_leaf() or self.children[child_index] == None:
		# 	raise Exception("Attempted to call UCB on a nonexistent node!")

		# UCB = Q + c * P/(1+N)
		q = 0
		if self.children[child_index].N:
			q = (self.children[child_index].Vsum/self.children[child_index].N * self.game.player + 1) / 2 # map q -> [0,1]
		u = MCTSNode.EXPLORATION_CONSTANT * self.P[child_index] * math.sqrt(self.N) / (1 + self.children[child_index].N)
		ucb = q + u

		return ucb

	def get_nn_input(self):
		if self.is_terminal():
			return None
		
		if self.game.player == -1:
			return torch.from_numpy(-self.game.board).unsqueeze_(0)
		return torch.from_numpy(self.game.board).unsqueeze_(0)

	def evaluate_and_rollout(self, output_tensor=None):

		if(self.is_terminal()):
			self.backprop(self.game.get_reward())
			return

		# expand children + evaluate using NN
		children_game_objs = self.game.get_children()
		self.children = []
		for game in children_game_objs:
			if game:
				self.children.append(MCTSNode(self, game))
			else:
				self.children.append(None)

		self.Vsum = (float(output_tensor[7]) * 2 - 1) * self.game.player # from NN (map sigmoid into (-1,1))

		noise = np.random.dirichlet([.03] * 7)
		self.P = [ .75 * float(output_tensor[i]) + .25 * float(noise[i]) for i in range(7)]
		self.N = 1

		if self.parent and not self.disable_backprop:
			self.parent.backprop(self.Vsum)

	def backprop(self, v):
		self.Vsum += v
		self.N += 1
		if self.parent and not self.disable_backprop:
			self.parent.backprop(v)

	def is_leaf(self):
		return not self.children

	def best_move(self):
		max_child = None
		max_i = -1
		for i in range(len(self.children)):
			if max_child == None or (self.children[i] != None and self.children[i].N > max_child.N):
				max_child = self.children[i]
				max_i = i

		return max_i
	
	def rand_move(self, temperature):
		max_child = self.children[self.best_move()] # must normalize by maxchild.N^temperature, as this might be an extremely large number

		s = 0
		for child in self.children:
			if child:
				s += math.pow(child.N / max_child.N, 1/temperature)

		x = s * random.random()
		for i in range(len(self.children)):
			if self.children[i]:
				x -= math.pow(self.children[i].N / max_child.N, 1/temperature)
				if x <= 0: 
					return i
		
		raise Exception("Don't know how we got here...")

class MCTS:

	def __init__(self, root: Connect4):
		if not root:
			root = MCTSNode.DEFAULT_BOARD
		self.root = MCTSNode( None, root )

	def iterate_without_final_rollout(self):
		current = self.root

		while(True):
			if current.is_terminal() or current.is_leaf():
				return current
			
			max_child = None
			max_i = None
			for i in range(len(current.children)):
				child = current.children[i]
				if max_child == None or ( child != None and current.UCB(i) > current.UCB(max_i) ):
					max_child = child
					max_i = i
			
			current = max_child
	
	def move(self, child_index):
		self.root = self.root.children[child_index]
		self.root.disable_backprop = True

	def move_to_rand_move(self, temperature=None):
		if temperature == None:
			temperature = 1 if self.root.game.num_moves < 7 else 0
		
		if temperature == 0:
			self.move( self.root.best_move() )
		self.move( self.root.rand_move( temperature ) )

	def generate_data_set(self):
		# if not self.root.is_terminal:
		# 	raise "Generating data set from non-terminal state!"

		result = self.root.game.get_reward()
		
		boards = []

		current = self.root
		while True:
			# print(current.game)
			boards.append(torch.from_numpy( current.game.board.astype(float) ) )
			if current.parent:
				current = current.parent
			else:
				break

		X = torch.stack(boards)
		Y = torch.ones((X.shape[0])) * result

		return (X,Y)

class MCTSManager:
	def __init__(self, num, nn, evaluations_per = 100):
		self.mcts = []
		self.finished = []
		self.nn = nn
		for _ in range(num):
			self.mcts.append(MCTS(None))
		self.EVALUATIONS_PER = evaluations_per
		self.node_ptrs = [None] * num

	def iterate(self):

		input_tensors = []
		for i in range(len(self.mcts)):
			m = self.mcts[i]
			node = m.iterate_without_final_rollout()
			input_tensor = node.get_nn_input()

			if input_tensor != None:
				input_tensors.append(input_tensor)
				self.node_ptrs[i] = node
			else:
				self.node_ptrs[i] = None
				node.evaluate_and_rollout(None)
		
		if len(input_tensors) > 0:
			input_torch = torch.stack(input_tensors).float().to(device)
			with torch.no_grad():
				pred = self.nn(input_torch).cpu().numpy()
			

		index = 0
		for i in range(len(self.mcts)):
			m = self.mcts[i]
			if self.node_ptrs[i] != None:
				self.node_ptrs[i].evaluate_and_rollout( pred[index] )
				index += 1
		
	def run(self):

		while True:
			for _ in range(self.EVALUATIONS_PER):
				self.iterate()
			
			for m in self.mcts:
				m.move_to_rand_move()
			
			new_mcts = []
			for m in self.mcts:
				if m.root.is_terminal():
					self.finished.append(m)
				else:
					new_mcts.append(m)
			
			self.mcts = new_mcts
			
			if len(self.mcts) == 0:
				return

class MCTSEvaluator(MCTSManager):

	def __init__(self, num, nn1, nn2, evaluations_per = 100):
		super().__init__(num, None, evaluations_per)
		self.nn1 = nn1
		self.nn2 = nn2
		self.player = 1
		self.outcomes = 0
	
	def run(self):
		while True:
			
			if self.player == 1:
				self.nn = self.nn1
			else:
				self.nn = self.nn2

			for _ in range(self.EVALUATIONS_PER):
				self.iterate()

			for m in self.mcts:
				m.move_to_rand_move(0)
			
			new_mcts = []
			for m in self.mcts:
				if m.root.is_terminal():
					self.finished.append(m)
					self.outcomes += m.root.game.get_reward()
					# print(m.root.game)
				else:
					new_mcts.append(m)
			
			self.mcts = new_mcts
			
			if len(self.mcts) == 0:
				return self.outcomes

class Connect4NN(nn.Module):
	INTERNAL_CHANNELS = 64

	def __init__(self):
		super().__init__()
		
		self.first_conv = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=Connect4NN.INTERNAL_CHANNELS, kernel_size=(3,3), stride=1, padding="same"),
			nn.BatchNorm2d(Connect4NN.INTERNAL_CHANNELS),
			nn.ReLU()
		)

		self.layers = nn.ModuleList()
		self.relus = nn.ModuleList()

		for i in range(8):
			self.layers.append(nn.Sequential(
				nn.Conv2d(in_channels=Connect4NN.INTERNAL_CHANNELS, out_channels=Connect4NN.INTERNAL_CHANNELS, kernel_size=(3,3), stride=1, padding="same"),
				nn.BatchNorm2d(Connect4NN.INTERNAL_CHANNELS),
				nn.ReLU(),
				nn.Conv2d(in_channels=Connect4NN.INTERNAL_CHANNELS, out_channels=Connect4NN.INTERNAL_CHANNELS, kernel_size=(3,3), stride=1, padding="same"),
				nn.BatchNorm2d(Connect4NN.INTERNAL_CHANNELS)
			))
			self.relus.append(nn.ReLU())

		self.policy_head = nn.Sequential(
			nn.Conv2d(in_channels=Connect4NN.INTERNAL_CHANNELS, out_channels=2, kernel_size=(1,1), stride=1, padding="same"),
			nn.BatchNorm2d(2),
			nn.ReLU()
		)

		self.policy_fc = nn.Linear(in_features=84,out_features=7) # outputs policy
		
		self.value_head = nn.Sequential(
			nn.Conv2d(in_channels=Connect4NN.INTERNAL_CHANNELS, out_channels=2, kernel_size=(1,1), stride=1, padding="same"),
			nn.BatchNorm2d(2),
			nn.ReLU()
		)

		self.value_fc = nn.Sequential(
			nn.Linear(in_features=84, out_features=84),
			nn.ReLU(),
			nn.Linear(in_features=84, out_features=1),
			nn.Sigmoid()
		)

	def forward(self, x):
		x = self.first_conv(x)

		for layer in self.layers:
			x = layer(x)
		
		ph = self.policy_head(x)
		vh = self.value_head(x)

		ph = torch.flatten(ph,1)
		vh = torch.flatten(vh,1)

		# print(ph.shape, vh.shape)
		p = self.policy_fc(ph)
		v = self.value_fc(vh)

		# print(p.shape, v.shape)
		return torch.cat((p,v),dim=1)

# model1 = Connect4NN()
# model2 = Connect4NN()

# model1.eval()
# model2.eval()

# def my_func(m1, m2):

# 	m1 = m1.to(device)
# 	m2 = m2.to(device)

# 	print("START!")

# 	a = time.time()

# 	m = MCTSEvaluator(20, m1, m2, 100)
# 	outcomes = m.run()

# 	b = time.time()

# 	print("FINISHED! ", (b-a), outcomes)

if __name__ == '__main__':

	mp.set_start_method('spawn')

	processes = []

	for i in range(5):
		pass
		# processes.append(mp.Process(target=my_func, args=(model1,model2)))

	for process in processes:
		process.start()

	for process in processes:
		process.join()

	# with torch.no_grad():
	# 	print(model1( torch.zeros((1,1,6,7)).to(device) ))
	# 	print(model1( torch.zeros((1,1,6,7)).to(device) ))
	# 	print(model1( torch.zeros((1,1,6,7)).to(device) ))