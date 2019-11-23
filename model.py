import torch
import torch.nn as nn
import torch.nn.functional as F

n_classes = 20

class ConvUnit(nn.Module):
    def __init__(self, in_deg, out_deg, kernel=3):
        super(ConvUnit, self).__init__()
        self.conv = nn.Conv2d(in_deg, out_deg, kernel_size=kernel)
        self.norm = nn.BatchNorm2d(num_features=out_deg)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(2048, 1000)
        self.fc1bis = nn.Linear(1000, 500)
        self.fc2 = nn.Linear(500, n_classes)

        self.dropoutdense = nn.Dropout(0.6)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = self.dropoutdense(x)
        x = F.relu(self.fc1bis(x))
        x = self.dropoutdense(x)
        return self.fc2(x)

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()

        self.start_deg = 32
        deg = self.start_deg
        next_deg = self.start_deg * 2

        self.conv1 = ConvUnit(3, deg)
        self.conv2 = ConvUnit(deg, deg)
        self.conv3 = ConvUnit(deg, next_deg)

        deg = next_deg
        next_deg *= 2

        self.conv4 = ConvUnit(deg, next_deg)
        self.conv5 = ConvUnit(next_deg, next_deg)
        self.conv6 = ConvUnit(next_deg, next_deg)

        deg = next_deg
        next_deg *= 2

        self.conv7 = ConvUnit(deg, next_deg)
        self.conv8 = ConvUnit(next_deg, next_deg)
        self.conv9 = ConvUnit(next_deg, next_deg)

        # deg = next_deg
        # next_deg *= 2

        self.conv10 = ConvUnit(next_deg, next_deg)
        self.conv11 = ConvUnit(next_deg, next_deg)

        self.fc1 = nn.Linear(self.start_deg*8, 100)
        self.fc2 = nn.Linear(100, n_classes)

        self.dropoutconv1 = nn.Dropout2d(0.15)
        self.dropoutconv2 = nn.Dropout2d(0.25)
        self.dropoutconv3 = nn.Dropout2d(0.3)

        self.dropoutdense1 = nn.Dropout(0.4)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropoutconv1(x)
        x = self.conv3(x)
        x = self.dropoutconv1(x)
        x = F.max_pool2d(x, 2)

        x = self.conv4(x)
        x = self.dropoutconv2(x)
        x = self.conv5(x)
        x = self.dropoutconv2(x)
        x = self.conv6(x)
        x = self.dropoutconv2(x)
        x = F.max_pool2d(x, 3)

        x = self.conv7(x)
        x = self.dropoutconv3(x)
        x = self.conv8(x)
        x = self.dropoutconv3(x)
        x = self.conv9(x)
        x = self.dropoutconv3(x)
        x = F.max_pool2d(x, 3)

        x = self.conv10(x)
        x = self.dropoutconv3(x)
        x = self.conv11(x)
        x = self.dropoutconv3(x)
        x = F.avg_pool2d(x, 4)

        x = x.view(-1, self.start_deg*8)
        x = F.relu(self.fc1(x))
        x = self.dropoutdense1(x)
        return self.fc2(x)