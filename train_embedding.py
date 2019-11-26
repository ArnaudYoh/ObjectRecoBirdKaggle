import argparse
import os

import numpy as np
import torch
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

# Training settings
parser = argparse.ArgumentParser(description='RecVis A3 training script')
parser.add_argument('--data', type=str, default='bird_dataset_mix', metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=16, metavar='B',
                    help='input batch size for training (default: 16)')
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 1000)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--experiment', type=str, default='experiment', metavar='E',
                    help='folder where experiment outputs are located.')
args = parser.parse_args()
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)

# Create experiment folder
if not os.path.isdir(args.experiment):
    os.makedirs(args.experiment)

class_remapping = {0: 11, 1: 8, 2: 7, 3: 18, 4: 14, 5: 4, 6: 3, 7: 1, 8: 13, 9: 19, 10: 15,
                   11: 5, 12: 10, 13: 12, 14: 0, 15: 9, 16: 2, 17: 16, 18: 17, 19: 6}

features_train = np.load('./features/bird_dataset_feature_train.npy')
labels_train = np.load('./features/bird_dataset_label_train.npy')
features_val = np.load('./features/bird_dataset_feature_val.npy')
labels_val = np.load('./features/bird_dataset_label_val.npy')
features_test = np.load('./features/bird_dataset_feature_test.npy')

features_train_crop = np.load('./features/bird_dataset_crop_feature_train.npy')
features_val_crop = np.load('./features/bird_dataset_crop_feature_val.npy')


train_tensor_x = torch.stack([torch.Tensor(i) for i in features_train])
val_tensor_x = torch.stack([torch.Tensor(i) for i in features_val])

train_dataset = TensorDataset(train_tensor_x, Tensor(labels_train))
val_dataset = TensorDataset(val_tensor_x, Tensor(labels_val))

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)

val_loader = torch.utils.data.DataLoader(
    val_dataset,batch_size=args.batch_size, shuffle=False, num_workers=1)

# Neural network and optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py script
from model import Net

model = Net()

if use_cuda:
    print('Using GPU')
    model.cuda()
else:
    print('Using CPU')

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
# optimizer = optim.Adam(model.parameters(), lr=args.lr)

def train(epoch):
    model.train()
    avg_accuracy = 0
    avg_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        target = target.long()
        loss = criterion(output, target)
        avg_loss += loss
        loss.backward()
        optimizer.step()

        pred = output.data.max(1, keepdim=True)[1]
        correct = pred.eq(target.data.view_as(pred)).cpu().sum()
        accuracy = 100. * correct.item() / len(target)
        avg_accuracy += accuracy
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Average Loss: {:.6f}, Acc: {:.1f}%'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data.item() / args.batch_size, accuracy))
    print("Average Training Accuracy: {:0.1f}%".format(avg_accuracy / len(train_loader)))
    print("Average Training Loss: {:0.6f}".format(avg_loss / len(train_loader.dataset)))


def validation():
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        # sum up batch loss
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        target = target.long()
        validation_loss += criterion(output, target).data.item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    accuracy = 100. * correct / len(val_loader.dataset)
    print('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        validation_loss, correct, len(val_loader.dataset),
        accuracy))

    return accuracy, validation_loss


best_acc = 0
ok_loss = 10
best_loss = 10
lr = args.lr
for epoch in range(1, args.epochs + 1):
    # 50 gave superb results
    if epoch % 75 == 0:
        lr *= 0.1
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=args.momentum)
        # optimizer = optim.Adam(model.parameters(), lr=lr)
    train(epoch)
    curr_accuracy, curr_loss = validation()
    model_file = args.experiment + '/model_' + str(epoch) + '.pth'
    if epoch % 10 == 0:
        torch.save(model.state_dict(), model_file)
        print('Saved model to ' + model_file + '. You can run `python evaluate.py --model ' + \
              model_file + '` to generate the Kaggle formatted csv file')

    elif curr_accuracy > best_acc and curr_loss < best_loss + 0.03:
        best_acc = curr_accuracy
        ok_loss = curr_loss
        torch.save(model.state_dict(), args.experiment + '/model_' + str(epoch) + 'BEST.pth')
    elif curr_loss < best_loss:
        best_loss = curr_loss
        ok_loss = curr_loss
        torch.save(model.state_dict(), args.experiment + '/model_' + str(epoch) + 'BESTLoss.pth')

    print('\n')
