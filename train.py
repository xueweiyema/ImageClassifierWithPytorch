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

parser = argparse.ArgumentParser(description='Flowers Classifer')

parser.add_argument('data_directory', action="store")
parser.add_argument(
    '--save_dir', action="store", default='checkpoint_test.pth')
parser.add_argument('--arch', action="store", default='vgg19')
parser.add_argument('--learning_rate', action="store", default=0.001, type=int)
parser.add_argument('--hidden_units', action="store", default='4096', type=int)
parser.add_argument('--num_labels', action="store", default='102', type=int)
parser.add_argument('--epochs', action="store", default='20', type=int)
parser.add_argument('--gpu', action="store", default=True, type=bool)

args = parser.parse_args()


def build_model(arch, num_labels, hidden_units):
    model = eval('models.' + arch + '(pretrained=True)')

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.classifier[0].in_features

    classifier = nn.Sequential(
        nn.Linear(num_features, hidden_units),
        nn.ReLU(True),
        nn.Dropout(p=0.5),
        nn.Linear(hidden_units, hidden_units),
        nn.ReLU(True),
        nn.Dropout(p=0.5),
        nn.Linear(hidden_units, num_labels))

    model.classifier = classifier
    return model


def train(data_dir, epochs, model, learning_rate, save_dir, gpu):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = {
        'train':
        transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'valid':
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'test':
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'valid', 'test']
    }

    dataloaders = {
        x: utils.data.DataLoader(
            image_datasets[x], batch_size=32, shuffle=True)
        for x in ['train', 'valid', 'test']
    }
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        list(filter(lambda p: p.requires_grad, model.parameters())),
        lr=learning_rate,
        momentum=0.9)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=7, gamma=0.1)

    print_every = 10
    steps = 0

    if torch.cuda.is_available() and gpu:
        print("Using GPU")
        device = torch.device("cuda:0")
        model.to('cuda')
    else:
        print("Using CPU")
        device = torch.device("cpu")

    dataset_sizes = {
        x: len(dataloaders[x].dataset)
        for x in ['train', 'valid', 'test']
    }
    for e in range(epochs):

        print('Epoch {}/{}'.format(epochs, e + 1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                exp_lr_scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            steps = 0

            for inputs, labels in dataloaders[phase]:

                steps += 1
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                if steps % print_every == 0:
                    print(
                        "Epoch: {}/{}... ".format(e + 1, epochs),
                        "Loss:{:.4f}".format(
                            running_loss / (steps * inputs.size(0))),
                        "Acc:{:.4f}".format(running_corrects.double() /
                                            (steps * inputs.size(0))))

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss,
                                                   epoch_acc))

    # model.class_to_idx = image_datasets['train'].class_to_idx

    # checkpoint = {'class_to_idx': model.class_to_idx,'state_dict': model.state_dict()}

    # torch.save(checkpoint, 'checkpoint_test.pth')
    torch.save(model, save_dir)

    phase = 'test'

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloaders[phase]:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' %
          (100 * correct / total))


m = build_model(args.arch, args.num_labels, args.hidden_units)
train(args.data_directory, args.epochs, m, args.learning_rate, args.save_dir,args.gpu)