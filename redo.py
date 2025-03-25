import copy
import numpy as np
import random
import math
import time
import cProfile
import torch.multiprocessing as mp
import multiprocessing
import psutil
from colorama import Fore, Back, Style

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.optim import Adam, SGD

DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

class Connect4: 

	BOARD_WIDTH = 7
	BOARD_HEIGHT = 6
	IN_A_ROW = 4

	def __init__(self, board: np.array, player: int, num_moves=0, i=None, j=None):
		self._reward = 0
		self._player = player
		self._board = board
		self._num_moves = num_moves
		self._is_terminal = (num_moves == Connect4.BOARD_HEIGHT * Connect4.BOARD_WIDTH)

		if num_moves > 0:
			for (idir, jdir) in [(1,0), (0,1), (1,1), (1,-1)]:
				self._check_in_a_row(i,j,idir, jdir)
		
		self._children = None
		
	def _check_in_a_row(self, i,j, i_dir, j_dir):
		color = self._board[i][j]

		TOT = 1
		i1, j1 = i + i_dir, j + j_dir
		while i1 < Connect4.BOARD_HEIGHT and j1 < Connect4.BOARD_WIDTH and i1 >= 0 and j1 >= 0 and self._board[i1][j1] == color:
			TOT += 1
			i1, j1 = i1 + i_dir, j1 + j_dir

		i1, j1 = i - i_dir, j - j_dir
		while i1 < Connect4.BOARD_HEIGHT and j1 < Connect4.BOARD_WIDTH and i1 >= 0 and j1 >= 0 and self._board[i1][j1] == color:
			TOT += 1
			i1, j1 = i1 - i_dir, j1 - j_dir
		
		if TOT >= 4:
			self._is_terminal = True
			self._reward = color
			return color

		return 0
	
	def __str__(self):
		output_str = "\n---------------\n"
		for i in reversed(range(Connect4.BOARD_HEIGHT)):
			output_str += '|'
			for j in range(Connect4.BOARD_WIDTH):
				output_str += Fore.RED + 'L' + Fore.RESET if self._board[i][j] == 1 else ( Fore.BLUE + 'R' + Fore.RESET if self._board[i][j] == -1 else ' ')
				output_str += '|'
			output_str += "\n---------------\n"

		return output_str
	
	def children(self):
		if self._children != None:
			return self._children

		self._children = []
		for j in range(Connect4.BOARD_WIDTH):
			if self._board[Connect4.BOARD_HEIGHT-1][j]:
				self._children.append(None)
			else:
				
				i = 0
				while self._board[i][j]:
					i = i+1
				
				new_board = np.copy(self._board)
				new_board[i][j] = self._player

				child = Connect4(new_board,-self._player,self._num_moves+1,i,j)
				self._children.append(child)
		
		return self._children

	def reward(self):
		return self._reward

	def is_terminal(self):
		return self._is_terminal

	def player(self):
		return self._player

	def tensor_representation(self):
		# 5 channel representation of self
		return torch.from_numpy(
			np.stack((
				self._board == -1,
				self._board == 0,
				self._board == 1,
				self._board,
				self.player() * np.ones_like(self._board)
			))).float()

class Connect4NN(nn.Module):

	INTERNAL_CHANNELS = 32
	RES_BLOCKS = 8

	def __init__(self):
		super().__init__()

		self.init_conv = nn.Sequential(
			nn.Conv2d(5, Connect4NN.INTERNAL_CHANNELS, kernel_size=(3,3), padding="same"),
			nn.BatchNorm2d(Connect4NN.INTERNAL_CHANNELS),
			nn.ReLU()
		)

		self.blocks = nn.ModuleList( 
			[ ResidualBlock(Connect4NN.INTERNAL_CHANNELS) for i in range(Connect4NN.RES_BLOCKS) ] 
		)

		self.policy_head = nn.Sequential(
			nn.Conv2d(Connect4NN.INTERNAL_CHANNELS, 2, kernel_size=(1,1), padding="same"),
			nn.BatchNorm2d(2),
			nn.ReLU(),
			nn.Flatten(),
			nn.Linear(Connect4.BOARD_WIDTH * Connect4.BOARD_HEIGHT * 2, Connect4.BOARD_WIDTH) # policy logits
		)

		self.value_head = nn.Sequential(
			nn.Conv2d(Connect4NN.INTERNAL_CHANNELS, 2, kernel_size=(1,1), padding="same"),
			nn.BatchNorm2d(2),
			nn.ReLU(),
			nn.Flatten(),
			nn.Linear(Connect4.BOARD_WIDTH * Connect4.BOARD_HEIGHT * 2, Connect4.BOARD_WIDTH * Connect4.BOARD_HEIGHT * 2),
			nn.ReLU(),
			nn.Linear(Connect4.BOARD_WIDTH * Connect4.BOARD_HEIGHT * 2, 1),
			nn.Tanh()
		)

	def forward(self, x):
		output = self.init_conv(x)
		for block in self.blocks:
			output = block(output)
		
		policy = self.policy_head(output)
		value = self.value_head(output)

		return (policy, value)

class ResidualBlock(nn.Module):
	def __init__(self, num_channels):
		super().__init__()

		self.conv_1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding="same")
		self.batch_norm_1 = nn.BatchNorm2d(num_channels)
		self.conv_2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding="same")
		self.batch_norm_2 = nn.BatchNorm2d(num_channels)
		self.relu = nn.ReLU()

	def forward(self, x):
		output = self.relu(self.batch_norm_1(self.conv_1(x)))
		output = self.batch_norm_2(self.conv_2(output))
		return self.relu(output + x)

class MCTSBatchSelfPlayer:
	def __init__(self, batch_size, model: Connect4NN, evaluations_per):
		self.roots = [MCTSNode(None, Connect4(np.zeros((Connect4.BOARD_HEIGHT, Connect4.BOARD_WIDTH)), 1)) for i in range(batch_size)]
		self.finished_roots = []
		self.model = model
		self.evaluations_per = evaluations_per

	def iterate(self):
		stack_of_evaluations = []
		nodes_to_expand = []

		for root in self.roots:
			node_to_expand = root.traverse()
			
			tensor_rep = node_to_expand.evaluation_tensor_representation()
			if tensor_rep != None:
				stack_of_evaluations.append(tensor_rep)
				nodes_to_expand.append(node_to_expand)
			else:
				node_to_expand.expand() # terminal states can be expanded without evaluation

		if len(stack_of_evaluations) > 0:
			self.model.eval()
			with torch.no_grad():
				policy_logits, values = self.model(torch.stack(stack_of_evaluations).to(DEVICE))
				policy_logits = policy_logits.cpu()
				values = values.cpu()
			
			for i in range(len(nodes_to_expand)):
				node_to_expand = nodes_to_expand[i]
				policy = torch.softmax(policy_logits[i],0).numpy()

				node_to_expand.expand(policy, float(values[i]))

	def move(self):

		new_roots = []

		for i in range(len(self.roots)):
			root = self.roots[i]
			root.toggle_backprop(False)
			self.roots[i] = root._children[root.pick_child_index( root.training_temperature() )]

			if self.roots[i].is_terminal():
				self.finished_roots.append(self.roots[i])
			else:
				new_roots.append(self.roots[i])
		
		self.roots = new_roots

	def run(self):
		while True:
			for _ in range(self.evaluations_per):
				self.iterate()

			input("MOVING")
			for root in self.roots:
				print(root)
			self.move()

			if len(self.roots) == 0:
				return

class MCTSNode:

	EXPLORATION_CONSTANT = 1
	D_ALPHA = .3
	EPS_NOISE = .25

	def __init__(self, parent, game: Connect4):
		self._game = game
		self._parent = parent

		self._W = 0
		self._N = 0
		self._P = []

		self._children = None
		self._disable_backprop = False

	def backprop(self, value):
		self._W += value
		self._N += 1

		if self._parent != None and not self._disable_backprop:
			self._parent.backprop(value)
	
	def UCB_of_child(self, child_index):
		# how much we want to go to that child from self
		q = 0
		if self._children[child_index]._N > 0:
			# win rate from my perspective
			q = self._children[child_index]._W / self._children[child_index]._N * self._game.player() 
		
		u = MCTSNode.EXPLORATION_CONSTANT * self._P[child_index] * math.sqrt(self._N) / (1 + self._children[child_index]._N)

		return q + u

	def largest_UCB_index(self):
		max_index = 0
		for i in range(len(self._children)):
			if self._children[max_index] == None or ( self._children[i] != None and self.UCB_of_child(max_index) < self.UCB_of_child(i) ):
				max_index = i
		return max_index

	def expand(self, predicted_policy=None, predicted_value=None):
		if self.is_terminal():
			self.backprop(self._game.reward())
		else:

			noise = np.random.dirichlet([ MCTSNode.D_ALPHA ] * Connect4.BOARD_WIDTH)

			self._P = predicted_policy * (1-MCTSNode.EPS_NOISE) + noise * MCTSNode.EPS_NOISE
			self.backprop(predicted_value)

			game_instances = self._game.children()
			self._children = []
			for game_instance in game_instances:
				if game_instance == None:
					self._children.append(None)
				else:
					self._children.append(MCTSNode( self, game_instance ))

	def is_terminal(self):
		return self._game.is_terminal()
	
	def is_leaf(self):
		return self._children == None
	
	def toggle_backprop(self, backprop):
		self._disable_backprop = not backprop
	
	def pick_child_index(self, temperature):

		if temperature == 0:
			# pick max child
			max_index = 0
			for i in range(len(self._children)):
				if self._children[max_index] == None or ( self._children[i] != None and self._children[max_index]._N < self._children[i]._N ):
					max_index = i
			return max_index
		else:
			max_child = self._children[self.pick_child_index(0)]

			s = 0
			for child in self._children:
				if child != None:
					s += math.pow( child._N / max_child._N, 1/temperature )
			
			r = random.random() * s

			for i in range(len(self._children)):
				child = self._children[i]
				if child != None:
					r -= math.pow( child._N / max_child._N, 1/temperature )

					if r <= 0:
						return i

	def traverse(self):
		current = self
		while(True):

			if current.is_terminal() or current.is_leaf():
				return current
			
			current = current._children[current.largest_UCB_index()]

	def evaluation_tensor_representation(self):
		if self.is_terminal():
			return None
		
		return self._game.tensor_representation()

	def output_tensor_representation(self, reward):
		Ns = np.array([child._N if child != None else 0 for child in self.children])
		Ns /= Ns.sum()
		return (self._game.tensor_representation(), torch.from_numpy(Ns).float(), torch.tensor([reward]).float())

	def training_temperature(self):
		return 0 if self._game._num_moves > 5 else 1

	def __str__(self):
		my_str = "--- MCTS Node ---\n"
		my_str += self._game.__str__()
		my_str += "N: " + str(self._N) + ", Q: " + str(self._W/self._N if self._N > 0 else 0) + "\n"

		if not self.is_leaf():
			for i in range(len(self._children)):
				child = self._children[i]
				if child != None:
					my_str += " Child N: " + str(child._N) + ", Q: " + str(child._W/child._N if child._N > 0 else 0) + ", P: " + str(self._P[i]) + " UCB: " + str(self.UCB_of_child(i)) + "\n"
		
		my_str += "--- MCTS Node End ---"
		return my_str

class MCTSOneSidedBatchComparator:
	def __init__(self, batch_size, model1: Connect4NN, model2: Connect4NN, evaluations_per):
		self.m1 = MCTSBatchSelfPlayer(batch_size, model1,evaluations_per)
		self.m2 = MCTSBatchSelfPlayer(batch_size, model2,evaluations_per)

		self.evaluations_per = evaluations_per

		self.outcomes = 0

	def run(self):
		while True:
			for _ in range(self.evaluations_per):
				self.m1.iterate()
				self.m2.iterate()
			
			self.move()

			if len(self.m1.roots) == 0:
				return self.outcomes

	def move(self):
		new_roots_1 = []
		new_roots_2 = []

		for i in range(len(self.m1.roots)):

			root1 = self.m1.roots[i]
			root2 = self.m2.roots[i]
			root1.toggle_backprop(False)
			root2.toggle_backprop(False)

			if root1._game.player() == 1:
				move_index = root1.pick_child_index( 0 )
			else:
				move_index = root2.pick_child_index( 0 )

			self.m1.roots[i] = root1._children[move_index]
			self.m2.roots[i] = root2._children[move_index]

			if self.m1.roots[i].is_terminal():
				self.outcomes += self.m1.roots[i]._game.reward()
			elif self.m2.roots[i].is_terminal():
				raise Exception("Not here!")
			else:
				new_roots_1.append(self.m1.roots[i])
				new_roots_2.append(self.m2.roots[i])
		
		self.m1.roots = new_roots_1
		self.m2.roots = new_roots_2
			
def MCTSCompare(batch_size, model1: Connect4NN, model2: Connect4NN, evaluations_per):
	m1 = MCTSOneSidedBatchComparator(batch_size,model1,model2,evaluations_per//2)
	m2 = MCTSOneSidedBatchComparator(batch_size,model2,model1,evaluations_per//2)

	a = m1.run()
	b = m2.run()
	
	return a-b

model = Connect4NN()
model.load_state_dict(torch.load('data/test.pth'))
model.to(DEVICE)

# model2 = Connect4NN()
# model2.to(DEVICE)

# g = Connect4(np.zeros((6,7)),1)
# print(g.tensor_representation())
# boards = [g.tensor_representation()] * 10
# a = torch.stack(boards).to(DEVICE)
# print(a.shape)
# output = (model(a))
# print(output)

a = MCTSBatchSelfPlayer(10,model,100)
a.run()

# m = MCTSBatchSelfPlayer(10, model, 10)
# m.run()
# for i in range(20):
# 	print(MCTSCompare(20, model2, model, 50))

# model.to(device)

# train_dataset = TensorDataset(X, Y, Z)
# train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=1)

# opt = torch.optim.Adam(model.parameters(), lr=0.0001)
# opt.load_state_dict(torch.load('data/testopt.pth'))

# cross = nn.CrossEntropyLoss()
# mse = nn.MSELoss()

# model.eval()

# for e in range(20):
# 	totalTrainLoss = 0

# 	for (x, y, z) in train_dataloader:
# 		(x, y, z) = (x.float().to(device), y.float().to(device), z.unsqueeze_(1).float().to(device))

# 		(policy_pred, value_pred) = model(x)

# 		print(x)
# 		print(policy_pred, value_pred)
# 		input()

	# 	loss = cross(policy_pred, y) + mse(value_pred, z)

	# 	opt.zero_grad()
	# 	loss.backward()
	# 	opt.step()

	# 	totalTrainLoss += loss
	
	# print(Fore.BLUE + "Training epoch " + str(e) + " completed. Training loss: " + str(float(totalTrainLoss)) )

# torch.save(model.state_dict(), "data/test.pth")
# torch.save(opt.state_dict(), "data/testopt.pth")


# DATA GENERATION
# Xs = []
# Ys = []
# Zs = []

# for i in range(500):
# 	game = Connect4(np.zeros((6,7)),1)

# 	while True:
# 		terminated = False
# 		children = game.children()
# 		for i in range(len(children)):
# 			child = children[i]
# 			if child != None and child.is_terminal():
# 				Y = [0] * len(children)
# 				Y[i] = 1
# 				Z = game.player()
# 				Xs.append(game.tensor_representation())
# 				Ys.append(torch.tensor(Y).float())
# 				Zs.append(Z)
# 				terminated = True
			
# 		if terminated:
# 			break

# 		children = [ child for child in children if child != None ]
# 		rand_i = random.randint(0,len(children)-1)
# 		game = children[rand_i]

# X = torch.stack(Xs)
# Y = torch.stack(Ys)
# Z = torch.tensor(Zs)

# torch.save(X, "data/Xs.pt")
# torch.save(Y, "data/Ys.pt")
# torch.save(Z, "data/Zs.pt")



