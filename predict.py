import os
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms, models
import torchvision
from torch import nn, optim, utils
import torch.utils.data as data
from PIL import Image
import numpy as np
import argparse
import json

parser = argparse.ArgumentParser(description='Flowers Classifer')

parser.add_argument('image_path', action='store')
parser.add_argument(
    '--checkpoint', action='store', default='checkpoint_test.pth')
parser.add_argument('--top_k', action='store', default=5, type=int)
parser.add_argument(
    '--category_names', action='store', default='cat_to_name.json')
parser.add_argument('--gpu', action='store', default=True, type=bool)

args=parser.parse_args()

def predict(category_names, image_path, checkpoint, top_k, gpu):
    with open('cat_to_name.json', 'r') as f:

        cat_to_name = json.load(f)

        cat_to_name_dict = dict(
            (int(key), value) for (key, value) in cat_to_name.items())

        dict_sorted = sorted(cat_to_name_dict.items(), key=lambda d: d[0])

        flowers_name = [value for key, value in dict_sorted]

    model = torch.load(checkpoint)

    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    im = Image.open(image_path)
    im = data_transforms(im).float()

    np_Image = np.array(im)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_Image = (np.transpose(np_Image, (1, 2, 0)) - mean) / std
    np_Image = np.transpose(np_Image, (2, 0, 1))
    np_Image = torch.autograd.Variable(torch.FloatTensor(np_Image), requires_grad=True)
    np_Image = np_Image.unsqueeze(0)

    if torch.cuda.is_available() and gpu:
        print("Using GPU")
        device = torch.device("cuda:0")
        model.cuda()
        np_Image = np_Image.cuda()
    else:
        print("Using CPU")
        device = torch.device("cpu")
    
    model.eval()

    result = model(np_Image).topk(top_k)
    
    probs = torch.nn.functional.softmax(result[0].data, dim=1).cpu().numpy()[0]
    classes = result[1].data.cpu().numpy()[0]

#     labels = list(cat_to_name.values())
    classes = [flowers_name[x] for x in classes]
    model.train(mode=model.training)

    return probs,classes

probs,classes=predict(args.category_names,args.image_path,args.checkpoint,args.top_k,args.gpu)
print(probs,classes)