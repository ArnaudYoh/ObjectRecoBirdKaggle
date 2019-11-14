import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 20 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=4)
        self.conv1bis = nn.Conv2d(16, 16, kernel_size=4)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=4)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=4)
        self.conv3bis = nn.Conv2d(32, 64, kernel_size=4)

        self.fc1 = nn.Linear(576, 40)
        self.fc2 = nn.Linear(40, nclasses)

        self.dropout_conv = nn.Dropout(0.35)
        self.dropout_lin = nn.Dropout(0.55)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout_conv(x)
        x = F.relu(F.max_pool2d(self.conv1bis(x), 2))
        x = self.dropout_conv(x)

        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = self.dropout_conv(x)
        x = F.relu(self.conv3(x))
        x = self.dropout_conv(x)
        x = F.relu(F.max_pool2d(self.conv3bis(x), 4))
        x = self.dropout_conv(x)

        x = x.view(-1, 576)
        x = torch.tanh(self.fc1(x))
        x = self.dropout_lin(x)

        return self.fc2(x)



#0.0912 35
