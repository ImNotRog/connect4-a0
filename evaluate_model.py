import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.optim import Adam
from c4 import Connect4NN

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

model = Connect4NN()

torch.load("c4data/model.pth")
model.load_state_dict(torch.load('c4data/model.pth', weights_only=True))

model.to(device)

# model_input = torch.tensor([
#          [0, 0, 1, 1, 1, 0, 0],
#          [0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 0]]).float().unsqueeze_(0).unsqueeze_(0)

# pred = model(model_input)
# print(pred)

def lossFn(output, target):
    logit_loss = nn.functional.cross_entropy(output[:, 0:7], target[:, 0:7])
    v_loss = nn.functional.binary_cross_entropy(output[:,7], target[:, 7])

    print(v_loss, logit_loss)
    return v_loss + logit_loss

# z = torch.tensor([0,100,0,0,0,100,0, 1]).float().unsqueeze_(0).expand(10,8)
# a = torch.tensor([0,.5,0,0,0,.5,0, 1]).float().unsqueeze_(0).expand(10,8)

# print(torch.softmax(pred[:, 0:7], dim=1))
# print(z)
# print(a)
# print(lossFn(pred, a))

# lossFn(z,a)
BATCH_SIZE = 64

X_tensor = torch.load("c4data/X.pt")
Y_tensor = torch.load("c4data/Y.pt")

train_dataset = TensorDataset(X_tensor, Y_tensor)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)

for (x, y) in train_dataloader:
    (x, y) = (x.to(device), y.to(device))

    pred = model(x)

    loss = lossFn(pred, y)
    