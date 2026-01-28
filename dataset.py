import os
import cv2
import numpy as np
import torch
import torch.utils.data

class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):

        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = []
        for i in range(self.num_classes):
            mask_path = os.path.join(self.mask_dir, str(i), img_id + self.mask_ext)
            if not os.path.exists(mask_path):

                mask_path = os.path.join(self.mask_dir, img_id + self.mask_ext)
            
            mask_item = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask.append(mask_item[..., None])
        
        mask = np.dstack(mask)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        if img.dtype == np.uint8:
            img = img.astype('float32') / 255.0

        mask = mask.astype('float32')
        mask[mask > 0] = 1.0

        if len(img.shape) == 3:
            img = img.transpose(2, 0, 1)
        if len(mask.shape) == 3:
            mask = mask.transpose(2, 0, 1)

        return img, mask, {'img_id': img_id}