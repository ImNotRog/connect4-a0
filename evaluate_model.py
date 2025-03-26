import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.optim import Adam
from redo import Connect4NN, Connect4, MCTSNode, MCTSBatchSelfPlayer, DEVICE
import numpy as np

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

model = Connect4NN()
model.load_state_dict(torch.load('models/best_model.pth', weights_only=True))
model.to(DEVICE)

class SinglePlayer:
	def __init__(self, model):
		self.root = MCTSNode(None, Connect4(np.zeros((Connect4.BOARD_HEIGHT, Connect4.BOARD_WIDTH)), 1))
		self.model = model
	
	def iterate(self):
		node_to_expand = self.root.traverse()
			
		tensor_rep = node_to_expand.evaluation_tensor_representation()
		if tensor_rep != None:

			self.model.eval()
			with torch.no_grad():
				policy_logits, values = self.model(torch.stack( (tensor_rep,) ).to(DEVICE))
				policy_logits = policy_logits.cpu()
				values = values.cpu()

			policy = torch.softmax(policy_logits[0],0).numpy()

			node_to_expand.expand(policy, float(values[0]))

		else:
			node_to_expand.expand() # terminal states can be expanded without evaluation

	def automove(self):
		self.root.toggle_backprop(False)
		self.root = self.root._children[self.root.pick_child_index( 0 )]

	def move(self, i):
		self.root = self.root._children[i]

	def __str__(self):
		return self.root._game.__str__() + "|0|1|2|3|4|5|6|\n"

	def is_terminal(self):
		return self.root.is_terminal()

m = SinglePlayer(model)

while not m.is_terminal():
	print(m)

	for i in range(100):
		m.iterate()

	i = int(input("Make your move! [0-6]:"))

	m.move(i)

	print(m)

	if m.is_terminal():
		break

	for i in range(100):
		m.iterate()

	print("Automoving.")
	m.automove()

print(m)
# current = MCTSNode(None, Connect4(np.zeros((6,7)),1))
# current.

# pred = model(c.tensor_representation().unsqueeze_(0))
# print(pred)
# print(next(model.parameters())[0])


