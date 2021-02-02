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

# import pdb

# pdb.set_trace()



os.environ['CUDA_VISIBLE_DEVICES']='2'
# 规定程序在哪个GPU上运行，最后一个数字代表GPU序号
print('PID:{}'.format(os.getpid()))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Trainings
def train(trainloader, net, criterion, optimizer, epoch):
    global device

    print('\nEpoch: %d' % epoch)
    net.train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    correct = 0
    total = 0
    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        data_time.update(time.time() - end)

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), inputs.size(0))
        
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        acc=100.*correct/total

        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % 20 == 0:
            output_log  = '[epoch:{epoch}] (iter:{batch}/{size}) Batch_time: {bt:.3f}s | TOTAL_time: {tt:.0f}min | ETA_time: {eta:.0f}min | Loss: {loss:.4f} | Acc: {acc: .4f}({correct_num}/{total_num})'.format(
                epoch =epoch, 
                batch=batch_idx,
                size=len(trainloader),
                bt=batch_time.avg,
                tt=batch_time.avg * batch_idx / 60.0,
                eta=batch_time.avg * (len(trainloader) - batch_idx) / 60.0,
                loss=losses.avg,
                acc=acc, 
                correct_num=correct, 
                total_num=total)
            print(output_log)
            sys.stdout.flush()
    return (losses.avg, acc, correct, total)

def adjust_learning_rate(args, optimizer, epoch):
    global state
    if epoch in args.schedule:
        args.lr = args.lr * 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

def save_checkpoint(state, checkpoint='checkpoint', filename='3d_checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

# def my_save_checkpoint(state, epoch, checkpoint='checkpoint', file_name='checkpoint.pth.tar'):
#     print('3d-resnet <==> Save checkpoint - epoch {} <==> Begin'.format(epoch))
#     file_name = 'epoch_'+str(epoch)+'_3d_checkpoint.pth.tar'
#     file_path = os.path.join(checkpoint, file_name)
#     torch.save(state, file_path)
#     print('3d-resnet <==> Save checkpoint - epoch {} <==> Done'.format(epoch))

def main(args):
    global device
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    if args.checkpoint == '':
        args.checkpoint = "checkpoints/%s_%s_L2_D64_bs_%d_ep_%d"%(args.dataset_mode, args.arch, args.batch_size, args.n_epoch)
    
    print ('checkpoint path: %s' %args.checkpoint)
    print ('init lr: %.8f' %args.lr)
    print ('schedule: ', args.schedule)
    sys.stdout.flush()

    if not os.path.isdir(args.checkpoint):
        os.makedirs(args.checkpoint)

    # Data
    print('==> Preparing data..')
    trainloader, testloader  = load_data(args)

    if args.dataset_mode == "CIFAR10":
        num_classes = 10
    elif args.dataset_mode == "CIFAR100":
        num_classes = 100
    elif args.dataset_mode == "IMAGENET":
        num_classes = 1000
    print('num_classes: ', num_classes)
    # transform_train = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])

    # trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    # trainloader = torch.utils.data.DataLoader(
    #     trainset, 
    #     batch_size=args.batch_size, 
    #     shuffle=True, 
    #     num_workers=2)
    
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
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    if args.pretrain:
        print('==> Using pretrained model.')
        assert os.path.isfile(args.pretrain), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.pretrain)
        net.load_state_dict(checkpoint['state_dict'])
    elif args.resume:
        print('==> Resuming from checkpoint.')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.resume)
        net.load_state_dict(checkpoint['net'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['acc']        
    else:
        print('==> Training from scratch.')

    for epoch in range(start_epoch, args.n_epoch):
        adjust_learning_rate(args, optimizer, epoch)
        print('\nEpoch: [%d | %d] LR: %f' % (epoch, args.n_epoch, optimizer.param_groups[0]['lr']))

        train_loss, train_acc, train_correct, train_total = train(trainloader, net, criterion, optimizer, epoch)
        # 保存最好的模型
        if train_acc > best_acc:
            print('Saving checkpoint ...')
            state = {
                'net'        : net.state_dict(),
                'optimizer'  : optimizer.state_dict(),
                'epoch'      : epoch,
                'acc'        : train_acc,
                'lr'         : args.lr,
            }
            filename = "Best_{}_{}_L2_D8_checkpoint.pth.tar".format(args.dataset_mode, args.arch)
            print("saveing bestModel...")
            save_checkpoint(state=state, checkpoint=args.checkpoint, filename=filename)
            best_acc = train_acc
        
        # 保存最新的模型
        state = {
            'net'        : net.state_dict(),
            'optimizer'  : optimizer.state_dict(),
            'epoch'      : epoch,
            'acc'        : train_acc,
            'lr'         : args.lr,
            }
        filename = "trainLatest_{}_{}_L2_D8_checkpoint.pth.tar".format(args.dataset_mode, args.arch)
        print("saveing latestModel...")
        save_checkpoint(state=state, checkpoint=args.checkpoint, filename=filename)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--arch', nargs='?', type=str, default='VGG3d')
    parser.add_argument("--dataset_mode", type=str, default="CIFAR10", 
                        help="(example: CIFAR10, CIFAR100), (default: CIFAR10)")
    parser.add_argument('--n_epoch', nargs='?', type=int, default=450, 
                        help='# of the epochs')
    parser.add_argument('--schedule', type=int, nargs='+', default=[150, 250, 350,],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--batch_size', nargs='?', type=int, default=128, 
                        help='Batch Size')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--lr', nargs='?', type=float, default=1e-1, 
                        help='Learning Rate')
    parser.add_argument('--resume', nargs='?', type=str, default=None,    
                        help='Path to previous saved model to restart from')
    parser.add_argument('--pretrain', nargs='?', type=str, default=None,    
                        help='Path to previous saved model to restart from')
    parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
    args = parser.parse_args()

    main(args)

