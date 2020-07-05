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
from PIL import Image
import cv2


CKPT_PATH = 'model.pth.tar'
N_CLASSES = 14
CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
DATA_DIR = './chest_xray/ChestX-ray14/images'
TEST_IMAGE_LIST = './chest_xray/ChestX-ray14/labels/auc_list.txt'
BATCH_SIZE = 64
THRESHOLD = 0.7


## Pysyft virtual workers have a limited memory to work
## Each worker can take a maximum of 4 images
hook = sy.TorchHook(torch)
image_names = []
with open(TEST_IMAGE_LIST, 'r') as f:
  for line in f:
    items = line.split()
    image_names.append(items[0])
workers = []
for i in range(len(image_names) // 3):
  worker = sy.VirtualWorker(hook, id=str(i))
  workers.append(worker)
print("Number of workers created = {}".format(len(workers)))

def main():

    cudnn.benchmark = True

    model = DenseNet121(N_CLASSES).cuda()

    ## getting the images list to be used for heatmap generation
    # image_names = []
    # with open(TEST_IMAGE_LIST, 'r') as f:
    #     for line in f:
    #         items = line.split()
    #         image_names.append(items[0])
    ##############################################################


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
    test_loader = sy.FederatedDataLoader(federated_dataset=test_dataset.federate(tuple(workers)), batch_size=BATCH_SIZE,
                             shuffle=False)

    # initialize the ground truth and output tensor
    gt = torch.FloatTensor()
    gt = gt.cuda()
    pred = torch.FloatTensor()
    pred = pred.cuda()

    # switch to evaluate mode
    model.eval()
    model_nosyft = model
    for i, (inp, target) in enumerate(test_loader):
        location = inp.location
        target = target.get().cuda()
        gt = torch.cat((gt, target), 0)
        inp = inp.get()
        bs, n_crops, c, h, w = inp.size()
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

    AUROCs, specificity, sensitivity, F1score = compute_AUCs(gt, pred)
    AUROC_avg = np.array(AUROCs).mean()
    print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
    for i in range(N_CLASSES):
        print('The AUROC of {} is {}'.format(CLASS_NAMES[i], AUROCs[i]))
        print('The specificity of {} is {}'.format(CLASS_NAMES[i], specificity[i]))
        print('The sensitivity of {} is {}'.format(CLASS_NAMES[i], sensitivity[i]))
        print('The F1 score of {} is {}'.format(CLASS_NAMES[i], F1score[i]))

    for arr in pred_np:
      indexes.append([index for index, val in enumerate(arr) if val == 1])
    for index, disease_array in enumerate(indexes):
        if len(disease_array) > 0:
            print(f'XRAY :: {index + 1} :: {[CLASS_NAMES[i] for i in disease_array]}')
        else:
            print(f'XRAY :: {index + 1} :: No Disease detected')
        print("Confidence of the 14 classes are {}".format(pred[i] * 100))
        # get_heat_map(model_nosyft, image_names[index], DATA_DIR)


def compute_AUCs(gt, pred):
    """Computes Area Under the Curve (AUC) from prediction scores.

    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.

    Returns:
        List of AUROCs of all classes.
    """
    AUROCs = []
    sensitivity = []
    specificity = []
    F1score = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()

    pred_np_final = pred.cpu().numpy()
    pred_np_final[pred_np_final < THRESHOLD] = 0
    pred_np_final[pred_np_final >= THRESHOLD] = 1
    indexes = []


    for i in range(N_CLASSES):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
        TP, FP, TN, FN = perf_measure(gt_np[:, i],pred_np_final[:, i])
        if (TP + FN == 0):
          print(f"issue with sensitivity {CLASS_NAMES[i]}")
          FN = 1
        if (TN + FP == 0):
          print(f"issue with specificity {CLASS_NAMES[i]}")
          FP = 1
        if (TP + FP == 0):
          print(f"issue with precision {CLASS_NAMES[i]}")
          FP = 1
        sensitivity.append(TP / (TP + FN))
        precision = TP/ (TP + FP)
        recall = TP / (TP + FN)
        if (precision + recall == 0):
          print(f"issue with F1score {CLASS_NAMES[i]}")
          recall = 1
        F1score.append(2 * ((precision * recall)/(precision + recall)))
        specificity.append(TN / (TN + FP))
    return AUROCs, specificity, sensitivity, F1score


def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(y_hat)): 
        if y_actual[i] == y_hat[i] == 1.0:
           TP += 1
        if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
           FP += 1
        if y_actual[i] == y_hat[i] == 0.0:
           TN += 1
        if y_hat[i] == 0 and y_actual[i] != y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)


def get_heat_map(model, image_name, DATA_DIR):
    test_model = model.densenet121.features
    test_model.eval()
    weights = list(test_model.parameters())[-2]
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    #---- Initialize the image transform - resize + normalize
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.TenCrop(224),
        transforms.Lambda
        (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda
        (lambda crops: torch.stack([normalize(crop) for crop in crops]))])
    
    #--------------------------------------------------------------------------------
    imageData = Image.open(os.path.join(DATA_DIR, image_name)).convert('RGB')
    imageData = transform(imageData)

    imageData_cuda = imageData.cuda()
    output = test_model(imageData_cuda)
    
    #---------- Heatmap Generation ---------------------------------------------------
    heatmap = None
    for j in range(10):
        for i in range(0, len(weights)):
            maps = output[j,i,:,:]
            if i == 0:
                heatmap = weights[i] * maps
            else:
                heatmap += weights[i] * maps
    #----------------------------------------------------------------------------------

    heatmap_np = heatmap.cpu().data.numpy()
    imOriginal = cv2.imread(os.path.join(DATA_DIR, image_name))
    imgOriginal = cv2.resize(imOriginal, (256, 256))


    cam = heatmap_np / np.max(heatmap_np)
    cam = cv2.resize(cam, (256, 256))
    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    img = heatmap * 0.5 + imgOriginal
    cv2.imwrite(f'./{image_name}',img)





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