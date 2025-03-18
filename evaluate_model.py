import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.optim import Adam
from c4 import Connect4NN

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

model = Connect4NN()

model.load_state_dict(torch.load('c4data/CURRENT_MODEL.pth', weights_only=True))

model_input = torch.tensor([[ 1,  1, 1, 0, 0, 0, 0],
         [ 0.,  -1.,  0., 0.,  0.,  0.,  0.],
         [ 0.,  -1,  0.,  0.,  0.,  0.,  0.],
         [ 0.,  -1.,  0.,  0.,  0.,  0.,  0.],
         [ 0.,  0,  0.,  0.,  0.,  0.,  0.],
         [ 0.,  0.,  0.,  0.,  0.,  0.,  0.]]).float().unsqueeze_(0).unsqueeze_(0)

pred = model(model_input)
print(pred)


Xprev = torch.load("c4data/X.pt")
Yprev = torch.load("c4data/Y.pt")

print(Xprev[1], Yprev[1])

