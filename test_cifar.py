#-*- coding: utf-8 -*-
'''Train CIFAR10 with PyTorch.'''
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import time
import argparse

from models import *
from preprocess import load_data

import numpy as np
import shutil
from utils import AverageMeter



os.environ['CUDA_VISIBLE_DEVICES']='3'
# 规定程序在哪个GPU上运行，最后一个数字代表GPU序号
print('PID:{}'.format(os.getpid()))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Trainings
def test(testloader, net, criterion):
    global device
    net.eval()
    
    batch_time = AverageMeter()
    losses = AverageMeter()

    correct = 0
    total = 0
    end = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            losses.update(loss.item(), inputs.size(0))
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            acc=100.*correct/total

            batch_time.update(time.time() - end)
            end = time.time()

            if (batch_idx+1) % 20 == 0:
                output_log  = '(iter:{batch}/{size}) fps: {fps:.2f} | Batch_time: {bt:.3f}s | TOTAL_time: {tt:.0f}min | ETA_time: {eta:.0f}min | Loss: {loss:.4f} | Acc: {acc: .4f}({correct_num}/{total_num})'.format(
                    batch=batch_idx,
                    size=len(testloader),
                    # fps=targets.size(0)/batch_time.avg,
                    fps=100./batch_time.avg,
                    bt=batch_time.avg,
                    tt=batch_time.avg * batch_idx / 60.0,
                    eta=batch_time.avg * (len(testloader) - batch_idx) / 60.0,
                    loss=losses.avg,
                    acc=acc, 
                    correct_num=correct, 
                    total_num=total)
                print(output_log)
                sys.stdout.flush()
    return (losses.avg, acc, correct, total)


def main(args):
    global device
    
    # best_acc = 0  # best test accuracy
    # start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # if not os.path.isdir(args.checkpoint):
    #     os.makedirs(args.checkpoint)

    # Data
    print('==> Preparing testdata..')
    trainloader, testloader = load_data(args)

    if args.dataset_mode == "CIFAR10":
        num_classes = 10
    elif args.dataset_mode == "CIFAR100":
        num_classes = 100
    elif args.dataset_mode == "IMAGENET":
        num_classes = 1000
    print('num_classes: ', num_classes)
    # transform_test = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])

    # testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    #Model
    print('==> Building model..')
    if args.arch == "resnet50_3d":
        net = resnet50_3d(num_classes)
    elif args.arch == "resnet50_3d_4":
        net = resnet50_3d_4(num_classes)
    elif args.arch == "resnet50_3d_34":
        net = resnet50_3d_34(num_classes)
    elif args.arch == "resnet50_3d_234":
        net = resnet50_3d_234(num_classes)
    elif args.arch == "VGG3d":
        net = VGG3d('VGG16', num_classes)
    # net = VGG('VGG19')
    # net = ResNet18()
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    # net = ShuffleNetV2(1)
    # net = EfficientNetB0()

    # for param in model.parameters():
    #     param.requires_grad = False

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    
    #load model from resume
    if args.resume:
        if os.path.isfile(args.resume):
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            net.load_state_dict(checkpoint['net'])
            epoch = checkpoint['epoch']
            train_acc = checkpoint['acc']
            print("Loaded checkpoint '{}' (epoch: {},train_acc: {})"
                .format(args.resume, epoch, train_acc))
            sys.stdout.flush()        
        else:
            print("No checkpoint found at '{}'".format(args.resume))
            sys.stdout.flush()

    test_loss, test_acc, test_correct, test_total = test(testloader, net, criterion)
    print("The accury of {}_L2_D8_ on cifar10 is {}({}/{})".format(args.arch, test_acc, test_correct, test_total))
    sys.stdout.flush()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Testing')
    parser.add_argument('--arch', nargs='?', type=str, default='VGG3d')
    parser.add_argument("--dataset_mode", type=str, default="CIFAR10", 
                        help="(example: CIFAR10, CIFAR100), (default: CIFAR10)")
    parser.add_argument('--batch_size', nargs='?', type=int, default=100, 
                        help='Batch Size')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--resume', nargs='?', type=str, 
                default="/share/home/math8/wpy/pytorch-cifar-master3d/checkpoints/CIFAR10_VGG3d_L2_D64_bs_128_ep_450/trainLatest_CIFAR10_VGG3d_L2_D64_checkpoint.pth.tar",    
                        help='Path to previous saved model to restart from')
    args = parser.parse_args()

    main(args)