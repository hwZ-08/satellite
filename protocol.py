import argparse
import os

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim

from models.resnet import ResNet18 
from models import ResNet18_cifar10
from utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='main test')
    parser.add_argument('--train_db_path', default='./data', help='the root path of the trainning data')
    parser.add_argument('--test_db_path', default='./data', help='the root path of the testing data')
    parser.add_argument('--dataset', default='cifar10', help='tain on [DATASET]')
    parser.add_argument('--wm_path', default='./data/trigger_set/', help='the root path of the wm set')
    parser.add_argument('--wm_lbl', default='labels-cifar.txt', help='random labels for triggers')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--lradj', type=int, default=20, help='multiple the lr by 0.1 every [LRADJ] epochs')
    parser.add_argument('--max_epochs', type=int, default=60, help='the maximun epochs')
    parser.add_argument('--batchsize', type=int, default=100, help='batchsize')
    parser.add_argument('--wm_batchsize', type=int, default=2, help='wm batchsize')
    parser.add_argument('--save_dir', default='./checkpoint', help='the root path of saved models')
    parser.add_argument('--save_model', default='model.pth', help='test model')
    parser.add_argument('--load_path', default='./checkpoint/ckpt.pth', help='pre-trained model, need flag resume')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--wm', action='store_true', help='train with wm')
    parser.add_argument('--log_dir', default='./log', help='the root path of log')
    parser.add_argument('--conf', default='config.txt', help='config file')
    parser.add_argument('--runname', default='train', help='the running program name')

    args = parser.parse_args()

    LOG_DIR = args.log_dir
    if not os.path.isdir(LOG_DIR):
        os.mkdir(LOG_DIR)
    logfile = os.path.join(LOG_DIR, 'log_' + str(args.runname) + '.txt')
    configfile = args.conf

    # save the configuration parameters
    with open(configfile, 'w') as f:
        for arg in vars(args):
            f.write(f'{arg}: {getattr(args, arg)}\n')

    start_epoch = 0   # start from epoch 0 or last checkpoint epoch
    model = ResNet18_cifar10()

    trainloader, testloader, n_classes = model.getdataloader(
        args.dataset, args.train_db_path, args.test_db_path, args.batchsize)

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model.net = model.net.to(device)
    ''' for mac
    if device == 'mps':
        cudnn.benchmark = True   # use cudnn to accelerate convolution 
        model.net = torch.nn.DataParallel(model.net, device_ids=range(torch.backends.mps.device_count()))
    '''

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    for epoch in range(start_epoch, start_epoch + args.max_epochs):
        # adjust learning rate 
        adjust_lr_rate(args.lr, optimizer, epoch, args.lradj)

        model.train(epoch, criterion, optimizer, logfile, trainloader, device)

        acc = model.test(criterion, logfile, testloader, device)
        print(f"Test acc: {acc:.3f}")

        if (epoch % 10 == 0):
            print("Saving...")
            model.savemodel(acc, epoch, device, args.save_dir)

    model.savemodel(acc, epoch, device, args.save_dir)
    