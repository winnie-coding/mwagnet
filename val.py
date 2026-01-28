import argparse
import os
from glob import glob
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml
from PIL import Image
from tqdm import tqdm
from albumentations import Resize, Normalize, Compose
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataset import Dataset
from metrics import indicators 
from utils import AverageMeter
from MWAG_Net_20260107 import MWAG_Net 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default=None, help='experiment name(e.g. mwagnet_isic2017)')
    parser.add_argument('--output_dir', default='outputs')
    args = parser.parse_args()
    return args

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def main():
    seed_torch()
    args = parse_args()

    config_path = os.path.join(args.output_dir, args.name, 'config.yml')
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-'*20)
    print(f"Testing Model: {args.name}")
    print(f"Dataset: {config['dataset']}")
    print('-'*20)


    model = MWAG_Net(config['input_channels'], config['num_classes'], 
                     config['wave_level'], model_size=config["model_size"])
    model = model.cuda()
    
    ckpt_path = os.path.join(args.output_dir, args.name, 'model.pth')
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()


    img_ext = '.jpg'
    if config['dataset'] == 'isic2017':
        mask_ext = '_segmentation.png'
    elif config['dataset'] == 'isic2018':
        mask_ext = '.png' 
    else:
        mask_ext = '.png'

    img_ids = sorted(glob(os.path.join(config['data_dir'], config['dataset'], 'images', '*' + img_ext)))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    _, test_img_ids = train_test_split(img_ids, test_size=0.2, random_state=config['dataseed'])

    test_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        Normalize(),
    ])

    test_dataset = Dataset(
        img_ids=test_img_ids,
        img_dir=os.path.join(config['data_dir'], config['dataset'], 'images'),
        mask_dir=os.path.join(config['data_dir'], config['dataset'], 'masks'),
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=config['num_classes'],
        transform=test_transform)

    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], 
                             shuffle=False, num_workers=config['num_workers'])

    avg_meters = {k: AverageMeter() for k in ['ji', 'dice', 'acc', 'sen', 'spe', 'hd95']}

    with torch.no_grad():
        for input, target, meta in tqdm(test_loader, total=len(test_loader)):
            input, target = input.cuda(), target.cuda()
            output = model(input)
            ji, dice, _, hd95, sen, spe, _, acc = indicators(output, target)

            avg_meters['ji'].update(ji, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))
            avg_meters['acc'].update(acc, input.size(0))
            avg_meters['sen'].update(sen, input.size(0))
            avg_meters['spe'].update(spe, input.size(0))
            avg_meters['hd95'].update(hd95, input.size(0))
            output_bin = torch.sigmoid(output).cpu().numpy()
            output_bin[output_bin >= 0.5] = 1
            output_bin[output_bin < 0.5] = 0

            save_path = os.path.join(args.output_dir, args.name, 'test_results')
            os.makedirs(save_path, exist_ok=True)
            for pred, img_id in zip(output_bin, meta['img_id']):
                pred_img = Image.fromarray((pred[0] * 255).astype(np.uint8), 'L')
                pred_img.save(os.path.join(save_path, f"{img_id}.png"))

    print(f"\n--- Final Test Results for {args.name} ---")
    print(f"Jaccard Index (JI): {avg_meters['ji'].avg:.4f}")
    print(f"Dice Coefficient:   {avg_meters['dice'].avg:.4f}")
    print(f"Accuracy (Acc):     {avg_meters['acc'].avg:.4f}")
    print(f"Sensitivity (Sen):  {avg_meters['sen'].avg:.4f}")
    print(f"Specificity (Spe):  {avg_meters['spe'].avg:.4f}")
    print(f"HD95 (mm):          {avg_meters['hd95'].avg:.4f}")

if __name__ == '__main__':
    main()