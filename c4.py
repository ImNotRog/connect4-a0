import numpy as np
import random
import math

import os
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.optim import Adam

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

    def __init__(self, parent, game: Connect4):
        self.game = game
        self.is_terminal = game.is_terminal
        self.children = False

        if not parent:
            self.parent = None
        else:
            self.parent = parent
        
        self.Vsum = 0
        self.N = 0
        self.P = [] # prior move probabilities

        self.disable_backprop = False

    def UCB(self, child_index):
        if self.is_leaf() or self.children[child_index] == None:
            raise Exception("Attempted to call UCB on a nonexistent node!")

        # UCB = Q + c * P/(1+N)
        q = 0
        if self.children[child_index].N:
            q = self.children[child_index].Vsum/self.children[child_index].N
        u = MCTSNode.EXPLORATION_CONSTANT * self.P[child_index] * math.sqrt(self.N) / (1 + self.children[child_index].N)
        ucb = q + u

        return ucb

    def evaluate_and_rollout(self):

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

        self.Vsum = 0 # from NN
        self.P = [1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7] # from NN 
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
        if self.is_terminal:
            raise Exception("Attempted to find best move at terminal state!")
        if not self.children:
            raise Exception("Attempted to move without evaluating children first!")
        
        max_child = None
        max_i = -1
        for i in range(len(self.children)):
            if max_child == None or (self.children[i] != None and self.children[i].N > max_child.N):
                max_child = self.children[i]
                max_i = i

        return max_i

class MCTS:

    def __init__(self, root: Connect4):
        self.root = MCTSNode( None, root )

    def iterate(self):
        current = self.root

        while(True):
            if current.is_terminal or current.is_leaf():
                current.evaluate_and_rollout()
                return
            
            max_child = None
            for child in current.children:
                if max_child == None or ( child != None and child.UCB() > max_child.UCB() ):
                    max_child = child
            
            current = child
    
    def move(self, child_index):
        if not self.root.children:
            raise Exception("Attempted to move without evaluating children first!")
        if not self.root.children[child_index]:
            raise Exception("Attempted to move to blank child!")

        self.root = self.root.children[child_index]
        self.root.disable_backprop = True

    def move_to_best_move(self):
        self.move( self.root.best_move() )

    def generate_data_set(self):
        # self.root
        pass

class Connect4NN(nn.Module):
	def __init__(self, classes):
		super().__init__()
		
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(5, 5))
		self.relu1 = nn.ReLU()
		self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

		self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5))
		self.relu2 = nn.ReLU()
		self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

		# initialize first (and only) set of FC => RELU layers
		self.fc1 = nn.Linear(in_features=800, out_features=500)
		self.relu3 = nn.ReLU()
		# initialize our softmax classifier
		self.fc2 = nn.Linear(in_features=500, out_features=classes)
		self.logSoftmax = nn.LogSoftmax(dim=1)

	def forward(self, x):
		# pass the input through our first set of CONV => RELU =>
		# POOL layers
		x = self.conv1(x)
		x = self.relu1(x)
		x = self.maxpool1(x)
		# pass the output from the previous layer through the second
		# set of CONV => RELU => POOL layers
		x = self.conv2(x)
		x = self.relu2(x)
		x = self.maxpool2(x)
		# flatten the output from the previous layer and pass it
		# through our only set of FC => RELU layers
		x = torch.flatten(x, 1)
		x = self.fc1(x)
		x = self.relu3(x)
		# pass the output to our softmax classifier to get our output
		# predictions
		x = self.fc2(x)
		output = self.logSoftmax(x)
		# return the output predictions
		return output

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")
    

