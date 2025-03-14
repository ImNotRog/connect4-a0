import numpy as np
import random
import math
import time
import cProfile


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

	def __init__(self, board:np.array, player: int):
		if board.shape != (Connect4.BOARD_HEIGHT, Connect4.BOARD_WIDTH):
			raise Exception("Invalid board dimensions!")
		self.board = np.copy(board) # 6 x 7 numpy matrix
		self.player = player # -1 or 1
		self.opp_board = - self.board
		self.children = False
		
		self.num_moves = int(sum(sum(abs( self.board ))))
		self.check_terminality()

	def check_terminality(self):
		self.is_terminal = True
		self.reward = 0

		for a in self.board:
			for b in a:
				if not b:
					self.is_terminal = False

		# horizontals
		for i in range(Connect4.BOARD_HEIGHT):
			for j in range(Connect4.BOARD_WIDTH - Connect4.IN_A_ROW+1):
				self.check_4_in_row(i,j,0,1)
		
		# verticals
		for i in range(Connect4.BOARD_HEIGHT - Connect4.IN_A_ROW + 1):
			for j in range(Connect4.BOARD_WIDTH):
				self.check_4_in_row(i,j,1,0)
				
		# diags
		for i in range(Connect4.BOARD_HEIGHT - Connect4.IN_A_ROW + 1):
			for j in range(Connect4.BOARD_WIDTH - Connect4.IN_A_ROW + 1):
				self.check_4_in_row(i,j,1,1)
		
		for i in range(Connect4.IN_A_ROW - 1, Connect4.BOARD_HEIGHT):
			for j in range(Connect4.BOARD_WIDTH - Connect4.IN_A_ROW + 1):
				self.check_4_in_row(i,j,-1,1)
			  
	def check_4_in_row(self, i,j, i_dir, j_dir):
		color = self.board[i][j]
		if not color:
			return 0

		in_a_row = True
		for k in range(1, Connect4.IN_A_ROW):
			if self.board[i + k * i_dir][j + k * j_dir] != color:
				in_a_row = False
		
		if in_a_row:
			self.is_terminal = True
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

				child = Connect4(new_board,-self.player)
				self.children.append(child)
		
		return self.children

class MCTSNode:

	EXPLORATION_CONSTANT = 3
	
	DEFAULT_BOARD = Connect4( np.zeros((6,7)), 1 )

	def __init__(self, parent, game: Connect4):
		self.game = game
		self.is_terminal = game.is_terminal
		self.children = False

		self.parent = parent
		# if not parent:
		# 	self.parent = None
		# else:
		# 	self.parent = parent
		
		self.Vsum = 0
		self.N = 0
		self.P = [] # prior move probabilities

		self.disable_backprop = False

	def UCB(self, child_index):
		# if self.is_leaf() or self.children[child_index] == None:
		# 	raise Exception("Attempted to call UCB on a nonexistent node!")

		# UCB = Q + c * P/(1+N)
		q = 0
		if self.children[child_index].N:
			q = self.children[child_index].Vsum/self.children[child_index].N * self.game.player
		u = MCTSNode.EXPLORATION_CONSTANT * self.P[child_index] * math.sqrt(self.N) / (1 + self.children[child_index].N)
		ucb = q + u

		return ucb

	def evaluate_and_rollout(self, neuralnet):

		if(self.is_terminal):
			self.backprop(self.game.reward)
			return

		# expand children + evaluate using NN
		children_game_objs = self.game.get_children()
		self.children = []
		for game in children_game_objs:
			if game:
				self.children.append(MCTSNode(self, game))
			else:
				self.children.append(None)

		if self.game.player == 1:
			result = neuralnet( torch.from_numpy(self.game.board).float().unsqueeze_(0).unsqueeze_(0) ) # TODO: batch
		else:
			result = neuralnet( torch.from_numpy(self.game.opp_board).float().unsqueeze_(0).unsqueeze_(0) )

		# result = [1,1,1,1,1,1,1,0]

		self.Vsum = (float(result[7]) * 2 - 1) * self.game.player # from NN (map sigmoid into (-1,1))
		self.P = [ float(i) for i in result[0:7]] # from NN TODO: Dirichlet noise
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

	def best_move(self): # MUST BE PROBABILISTIC
		# if self.is_terminal:
		# 	raise Exception("Attempted to find best move at terminal state!")
		# if not self.children:
		# 	raise Exception("Attempted to find best move without evaluating children first!")
		
		max_child = None
		max_i = -1
		for i in range(len(self.children)):
			if max_child == None or (self.children[i] != None and self.children[i].N > max_child.N):
				max_child = self.children[i]
				max_i = i

		return max_i
	
	def rand_move(self, temperature):
		# if self.is_terminal:
		# 	raise Exception("Attempted to find rand move at terminal state!")
		# if not self.children:
		# 	raise Exception("Attempted to find rand move without evaluating children first!")

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

	def __init__(self, root: Connect4, nn, evaluations_per = 100):
		if not root:
			root = MCTSNode.DEFAULT_BOARD
		self.root = MCTSNode( None, root )
		self.nn = nn

		self.EVALUATIONS_PER = evaluations_per

	def iterate(self):
		current = self.root

		while(True):
			if current.is_terminal or current.is_leaf():
				current.evaluate_and_rollout(self.nn)
				return
			
			max_child = None
			max_i = None
			for i in range(len(current.children)):
				child = current.children[i]
				if max_child == None or ( child != None and current.UCB(i) > current.UCB(max_i) ):
					max_child = child
					max_i = i
			
			current = max_child
	
	def move(self, child_index):
		# if not self.root.children:
		# 	raise Exception("Attempted to move without evaluating children first!")
		# if not self.root.children[child_index]:
		# 	raise Exception("Attempted to move to blank child!")

		self.root = self.root.children[child_index]
		self.root.disable_backprop = True

	def move_to_rand_move(self):
		self.move( self.root.rand_move( 1 if self.root.game.num_moves < 7 else 0.001 ) )

	def generate_data_set(self):
		# if not self.root.is_terminal:
		# 	raise "Generating data set from non-terminal state!"

		result = self.root.game.reward
		
		boards = []

		current = self.root
		while True:
			boards.append(torch.from_numpy( current.game.board.astype(float) ) )
			if current.parent:
				current = current.parent
			else:
				break
		
		X = torch.stack(boards)
		Y = torch.ones((X.shape[0])) * result

		return (X,Y)
	
	def run(self):
		while not self.root.is_terminal:
			for i in range(self.EVALUATIONS_PER): # 100 rollouts per move
				self.iterate()
		
			self.move_to_rand_move()
		
		return self.generate_data_set()

class Connect4NN(nn.Module):
	def __init__(self):
		super().__init__()
		
		self.first_conv = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3,3), stride=1, padding="same"),
			nn.BatchNorm2d(64),
			nn.ReLU()
		)

		self.layers = []
		self.relus = []
		for i in range(8):
			self.layers.append(nn.Sequential(
				nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=1, padding="same"),
				nn.BatchNorm2d(64),
				nn.ReLU(),
				nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=1, padding="same"),
				nn.BatchNorm2d(64)
			))
			self.relus.append(nn.ReLU())

		self.policy_head = nn.Sequential(
			nn.Conv2d(in_channels=64, out_channels=2, kernel_size=(1,1), stride=1, padding="same"),
			nn.BatchNorm2d(2),
			nn.ReLU()
		)

		self.policy_fc = nn.Linear(in_features=84,out_features=7) # outputs policy
		
		self.value_head = nn.Sequential(
			nn.Conv2d(in_channels=64, out_channels=2, kernel_size=(1,1), stride=1, padding="same"),
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

		ph = torch.flatten(ph)
		vh = torch.flatten(vh)

		p = self.policy_fc(ph)
		v = self.value_fc(vh)

		return torch.cat((p,v))

model = Connect4NN()

def my_func():
	print("START!")
	a = time.time()
	m = MCTS(None,model,100)
	m.run()
	b = time.time()

	print("FINISHED! ", b-a)

# for i in range(20):
# 	a = time.time()
# 	m = MCTS(None,model,50)
# 	m.run()
# 	b = time.time()

# 	print(b-a)

cProfile.run("my_func()",None,"tottime")
