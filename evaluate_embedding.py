import argparse

import numpy as np
import torch
from torch import Tensor
from model import Net

parser = argparse.ArgumentParser(description='RecVis A3 evaluation script')
parser.add_argument('--data', type=str, default='features', metavar='D',
                    help="folder where data is located. test_images/ need to be found in the folder")
parser.add_argument('--model', type=str, metavar='M',
                    help="the model file to be evaluated. Usually it is of the form model_X.pth")
parser.add_argument('--outfile', type=str, default='experiment/kaggle.csv', metavar='D',
                    help="name of the output csv file")

args = parser.parse_args()
use_cuda = torch.cuda.is_available()

state_dict = torch.load(args.model)
model = Net()
model.load_state_dict(state_dict)
model.eval()
if use_cuda:
    print('Using GPU')
    model.cuda()
else:
    print('Using CPU')

class_remapping = {0: 11, 1: 8, 2: 7, 3: 18, 4: 14, 5: 4, 6: 3, 7: 1, 8: 13, 9: 19, 10: 15,
                   11: 5, 12: 10, 13: 12, 14: 0, 15: 9, 16: 2, 17: 16, 18: 17, 19: 6}

test_data = np.load(args.data + '/bird_dataset_crop_feature_test.npy')
test_tensor = torch.stack([Tensor(x) for x in test_data])

file_names = open('./bird_dataset/test.txt', "r")

output_file = open(args.outfile, "w")
output_file.write("Id,Category\n")
for idx in range(len(test_tensor)):
    data = test_tensor[idx]
    f_n = file_names.readline()
    f_n = f_n.split('.')[0].split('/')[-1]
    if use_cuda:
        data = data.cuda()
    output = model(data)
    pred = output.data.max(0, keepdim=True)[1].item()
    pred = class_remapping[pred]
    output_file.write("%s,%d\n" % (f_n, pred))

output_file.close()

print("Succesfully wrote " + args.outfile + ', you can upload this file to the kaggle competition website')
