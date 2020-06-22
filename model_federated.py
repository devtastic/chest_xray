# encoding: utf-8

"""
The main CheXNet model implementation.
"""


import os
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from read_data import ChestXrayDataSet
from sklearn.metrics import roc_auc_score
import syft as sy
import re


## Initialize the pysyt parameters
hook = sy.TorchHook(torch)
bob = sy.VirtualWorker(hook, id='bob')
alice = sy.VirtualWorker(hook, id='alice')


CKPT_PATH = 'model.pth.tar'
N_CLASSES = 14
CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
DATA_DIR = './CheXNet/ChestX-ray14/images'
TEST_IMAGE_LIST = './CheXNet/ChestX-ray14/labels/new_list.txt'
BATCH_SIZE = 64
THRESHOLD = 0.9


def main():

    cudnn.benchmark = True

    # initialize and load the model
    model = DenseNet121(N_CLASSES).cuda()
    # model = torch.nn.DataParallel(model).cuda()

    if os.path.isfile(CKPT_PATH):
        print("=> loading checkpoint")
        checkpoint = torch.load(CKPT_PATH)
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = checkpoint['state_dict']
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        for key in list(state_dict.keys()):
          split_key = key.split('.')
          new_key = '.'.join(split_key[1:])
          state_dict[new_key] = state_dict[key]
          del state_dict[key]
        model.load_state_dict(state_dict)
        print("=> loaded checkpoint")
    else:
        print("=> no checkpoint found")

    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    test_dataset = ChestXrayDataSet(data_dir=DATA_DIR,
                                    image_list_file=TEST_IMAGE_LIST,
                                    transform=transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.TenCrop(224),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                    ]))
    test_loader = sy.FederatedDataLoader(federated_dataset=test_dataset.federate((bob, alice)), batch_size=BATCH_SIZE,
                             shuffle=False)

    # initialize the ground truth and output tensor
    gt = torch.FloatTensor()
    gt = gt.cuda()
    pred = torch.FloatTensor()
    pred = pred.cuda()

    # switch to evaluate mode
    model.eval()

    for i, (inp, target) in enumerate(test_loader):
        location = inp.location
        inp = inp.get()
        bs, n_crops, c, h, w = inp.size()
        # input_var = torch.autograd.Variable(torch.FloatTensor(inp.view(-1, c, h, w)).cuda(), volatile=True).send(location)
        input_var = inp.view(-1, c, h, w).cuda().send(location)
        model.send(location)
        output = model(input_var)
        output_mean = output.view(bs, n_crops, -1).mean(1)
        output_mean_data = output_mean.get()
        model.get()
        pred = torch.cat((pred, output_mean_data.data), 0)
    pred_np = pred.cpu().numpy()
    pred_np[pred_np < THRESHOLD] = 0
    pred_np[pred_np >= THRESHOLD] = 1
    indexes = []
    for arr in pred_np:
      indexes.append([index for index, val in enumerate(arr) if val == 1])
    for index, disease_array in enumerate(indexes):
      if len(disease_array) > 0:
        print(f'XRAY :: {index + 1} :: {[CLASS_NAMES[i] for i in disease_array]}')
      else:
        print(f'XRAY :: {index + 1} :: No Disease detected')


class DenseNet121(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x


if __name__ == '__main__':
    main()