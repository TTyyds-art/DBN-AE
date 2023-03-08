import torch
import numpy as np
from torch import nn, as_tensor
from torch.nn import functional as F
from torchsummary import summary


class StructNet(nn.Module):
    def __init__(self, stru_matrix=None):
        super(StructNet, self).__init__()
        if stru_matrix is None:
            stru_matrix = [32, 64, 256, 128]
        self.layers = np.array(stru_matrix)
        if len(self.layers.shape) > 1:
            self.layers = self.layers.squeeze()

        self.model = nn.ModuleList()
        for idx in range(len(self.layers) - 1):
            self.model.append(nn.Linear(self.layers[idx], self.layers[idx + 1]))

            if idx < len(self.layers) - 2:  # 最后一层不用ReLU
                self.model.append(nn.ReLU())

    def forward(self, x):
        pass

class DBN_C(nn.Module):
    def __init__(self, stru_matrix=None):
        super(DBN_C, self).__init__()
        if stru_matrix is None:
            stru_matrix = [32, 64, 256, 128]
        self.layers = np.array(stru_matrix)
        if len(self.layers.shape) > 1:
            self.layers = self.layers.squeeze()

        self.model = nn.ModuleList()
        for idx in range(len(self.layers) - 1):
            self.model.append(nn.Linear(self.layers[idx], self.layers[idx + 1]))

            if idx < len(self.layers) :
                self.model.append(nn.Sigmoid())

    def forward(self, x):

        for _, l in enumerate(self.model):
            # h_list.append(torch.mean(x, 0, True))
            x = l(x)
        return x


class CustomNet(StructNet):
    """Input: strcture matrix: [ ] and ouptut's range & middle values
       Then a MLP network is created"""

    def __init__(self, stru_matrix=None, range_x=1, middle_x=0):
        if stru_matrix is None:
            stru_matrix = [32, 64, 256, 128]
        self.truc_matrix = stru_matrix
        self.range_x, self.middle_x = as_tensor(range_x), as_tensor(middle_x)
        super(CustomNet, self).__init__(self.truc_matrix)
        # self.model.append(range_x * nn.Tanh() + middle_x )

    def forward(self, x):
        # h_list = []
        for _, l in enumerate(self.model):
            # h_list.append(torch.mean(x, 0, True))
            x = l(x)

        x = self.range_x * torch.tanh(x) + self.middle_x   # 这一层的输入不收集，因为没有可学习的变量
        # x = F.relu(x - (self.middle_x - 0.5 * self.range_x)) - F.relu(
            # x - (self.middle_x + 0.5 * self.range_x)) + self.middle_x - 0.5 * self.range_x

        return x


device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    my_net = CustomNet(stru_matrix=[32, 55, 88, 10]).to(device)
    x = torch.ones((1, 32)).to(device)
    y = my_net(x * 10000)
    print(summary(my_net, (1, 32), device=device))
    print(y.data)
