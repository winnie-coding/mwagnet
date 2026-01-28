import argparse
import os
from collections import OrderedDict
from glob import glob
import random
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml

from albumentations import RandomRotate90, Resize, HorizontalFlip, VerticalFlip, Normalize, Compose
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter

import losses
from dataset_0128 import Dataset
from metrics import iou_score, indicators
from utils import AverageMeter, str2bool
from MWAG_Net_20260107 import MWAG_Net 

LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='MWAGNet_ISIC_Experiment')
    parser.add_argument('--epochs', default=250, type=int)
    parser.add_argument('-b', '--batch_size', default=8, type=int)
    parser.add_argument('--dataseed', default=2981, type=int)
    
    # model
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int)
    parser.add_argument('--num_classes', default=1, type=int)
    parser.add_argument('--input_w', default=256, type=int)
    parser.add_argument('--input_h', default=256, type=int)
    parser.add_argument('--wave_level', default=2, type=int)
    parser.add_argument('--model_size', default='small', type=str, choices=['small', 'mid', 'large'])

    # loss & optimizer
    parser.add_argument('--loss', default='BCEDiceLoss', choices=LOSS_NAMES)
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'])
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    
    # dataset
    parser.add_argument('--dataset', default='isic2018', choices=['isic2017', 'isic2018'])      
    parser.add_argument('--data_dir', default='inputs')
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--num_workers', default=4, type=int)

    config = parser.parse_args()
    return config

def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(), 'iou': AverageMeter()}
    model.train()
    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input, target = input.cuda(), target.cuda()
        output = model(input)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0) 
        optimizer.step()

        iou, _, _ = iou_score(output, target)
        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))
        
        pbar.set_postfix(OrderedDict([('loss', avg_meters['loss'].avg), ('iou', avg_meters['iou'].avg)]))
        pbar.update(1)
    pbar.close()
    return OrderedDict([('loss', avg_meters['loss'].avg), ('iou', avg_meters['iou'].avg)])

def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(), 'iou': AverageMeter(), 'dice': AverageMeter()}
    model.eval()
    with torch.no_grad():
        for input, target, _ in val_loader:
            input, target = input.cuda(), target.cuda()
            output = model(input)
            loss = criterion(output, target)
            iou, dice, _ = iou_score(output, target)

            if loss.item() < 10.0: 
                avg_meters['loss'].update(loss.item(), input.size(0))
                avg_meters['iou'].update(iou, input.size(0))
                avg_meters['dice'].update(dice, input.size(0))
    return OrderedDict([('loss', avg_meters['loss'].avg), ('iou', avg_meters['iou'].avg), ('dice', avg_meters['dice'].avg)])

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark, cudnn.deterministic = True, False

def main():
    seed_torch()
    args = parse_args()
    config = vars(args)

    os.makedirs(f"{args.output_dir}/{args.name}", exist_ok=True)
    writer = SummaryWriter(f"{args.output_dir}/{args.name}")

    img_ext = '.jpg'
    mask_ext = '_segmentation.png' if args.dataset in ['isic2017', 'isic2018'] else '.png'
    img_ids = sorted(glob(os.path.join(args.data_dir, args.dataset, 'images', '*' + img_ext)))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    train_val_ids, test_ids = train_test_split(img_ids, test_size=0.2, random_state=args.dataseed)
    train_ids, val_ids = train_test_split(train_val_ids, test_size=0.125, random_state=args.dataseed)

    print(f"Split results: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")

    model = MWAG_Net(args.input_channels, args.num_classes, args.wave_level, model_size=args.model_size).cuda()
    criterion = losses.__dict__[args.loss]().cuda() if args.loss != 'BCEWithLogitsLoss' else nn.BCEWithLogitsLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    train_transform = Compose([RandomRotate90(p=0.5), HorizontalFlip(p=0.5), VerticalFlip(p=0.5), Resize(args.input_h, args.input_w), Normalize()])
    val_transform = Compose([Resize(args.input_h, args.input_w), Normalize()])

    train_loader = DataLoader(Dataset(train_ids, os.path.join(args.data_dir, args.dataset, 'images'), os.path.join(args.data_dir, args.dataset, 'masks'), img_ext, mask_ext, args.num_classes, train_transform),
                              batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(Dataset(val_ids, os.path.join(args.data_dir, args.dataset, 'images'), os.path.join(args.data_dir, args.dataset, 'masks'), img_ext, mask_ext, args.num_classes, val_transform),
                            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    best_iou = 0
    log = OrderedDict([('epoch', []), ('loss', []), ('iou', []), ('val_loss', []), ('val_iou', []), ('val_dice', [])])

    for epoch in range(args.epochs):
        print(f"Epoch [{epoch}/{args.epochs}]")
        train_log = train(config, train_loader, model, criterion, optimizer)
        val_log = validate(config, val_loader, model, criterion)
        scheduler.step()

        log['epoch'].append(epoch)
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])
        pd.DataFrame(log).to_csv(f"{args.output_dir}/{args.name}/log.csv", index=False)

        if val_log['iou'] > best_iou:
            best_iou = val_log['iou']
            torch.save(model.state_dict(), f"{args.output_dir}/{args.name}/model.pth")
            print(f"=> Saved best model (IoU: {best_iou:.4f})")

if __name__ == '__main__':
    main()