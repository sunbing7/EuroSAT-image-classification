#!/bin/env python3

import argparse
import os
import shutil

import torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, transforms
from tqdm import tqdm, trange

from dataset import *
from predict import predict


class State:
    # Keep some global state here (ex best accuracy on val)
    best_acc = 0
    writer: SummaryWriter = None
    normalization = None





def main(args):
    train_dl, val_dl, State.normalization, classes = get_train_loader(args.batch_size, args.workers)
    mean = State.normalization['mean']
    std = State.normalization['std']

    # create/load model, changing the head for our number of classes
    model = models.resnet50(pretrained=args.pretrained)
    #if args.pretrained:
    #    for param in model.parameters():
    #        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model = model.to(args.device)
    loss = nn.CrossEntropyLoss()

    params = model.fc.parameters() if args.pretrained else model.parameters()
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wd)

    State.writer = SummaryWriter()
    # display some examples in tensorboard
    images, labels = next(iter(train_dl))
    originals = images * std.view(3, 1, 1) + mean.view(3, 1, 1)
    State.writer.add_images('images/original', originals, 0)
    State.writer.add_images('images/normalized', images, 0)
    # writer.add_graph(model, images)

    for epoch in trange(args.epochs, desc="Epochs"):
        train_epoch(train_dl, model, loss, optimizer, epoch, args)
        truth, preds = predict(model, val_dl)

        torch.save(
            {'normalization': State.normalization, 'model_state': model.state_dict()},
            'weights/checkpoint.pt',
        )

        val_acc = (truth == preds).float().mean()
        State.writer.add_scalar('acc/val', val_acc, epoch * len(train_dl))
        if val_acc > State.best_acc:
            print(f"New best validation accuracy: {val_acc}")
            State.best_acc = val_acc
            shutil.copy('weights/checkpoint.pt', 'weights/best.pt')
    torch.save(model, 'weights/resnet50_eurosat.pth')

def train_epoch(train_dl, model, loss, optimizer, epoch, args):
    model.train()
    train_dl = tqdm(train_dl, "Train", unit="batch")
    for i, (images, labels) in enumerate(train_dl):
        images = images.to(args.device, non_blocking=True)
        labels = labels.to(args.device, non_blocking=True)

        preds = model(images)
        _loss = loss(preds, labels)
        acc = (labels == preds.argmax(1)).float().mean()

        optimizer.zero_grad()
        _loss.backward()
        optimizer.step()

        State.writer.add_scalar('loss/train', _loss, epoch * len(train_dl) + i)
        State.writer.add_scalar('acc/train', acc, epoch * len(train_dl) + i)


if __name__ == '__main__':

    def parse_bool(s: str):
        if s.casefold() in ['1', 'true', 'yes']:
            return True
        if s.casefold() in ['0', 'false', 'no']:
            return False
        raise ValueError()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-j',
        '--workers',
        default=4,
        type=int,
        metavar='N',
        help="Number of workers for the DataLoader",
    )
    parser.add_argument('--epochs', default=15, type=int, metavar='N', help="Epochs")
    parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N', help="Batch size")
    parser.add_argument(
        '--lr', '--learning-rate', default=0.0001, type=float, metavar='LR', help="Learning rate"
    )
    # parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
    parser.add_argument(
        '--wd', '--weight-decay', default=0, type=float, metavar='WD', help="Weight decay"
    )
    parser.add_argument(
        '--pretrained', default=True, type=parse_bool, help="Finetune a pre-trained model"
    )
    args = parser.parse_args()

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('weights', exist_ok=True)
    main(args)
