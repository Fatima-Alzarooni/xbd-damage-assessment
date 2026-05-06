# dataset class for xBD building damage assessment
# loads preprocessed pre/post/mask triples and applies augmentation on the fly
# used by all three models: Siamese U-Net, DeepLabV3+, SegFormer

import os
import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch
from augment import augment

class xBDDataset(Dataset):
    def __init__(self, preprocessed_dir, split='train'):

        # preprocessed_dir: path to the preprocessed folder
        # split: 'train', 'val', or 'test'
        self.split    = split
        self.is_train = split == 'train'
        
        self.pre_dir  = Path(preprocessed_dir) / split / 'pre'
        self.post_dir = Path(preprocessed_dir) / split / 'post'
        self.mask_dir = Path(preprocessed_dir) / split / 'masks'
        
        # get all filenames from post folder
        self.files = sorted([f.name for f in self.post_dir.iterdir()])

        # # drop no-damage-only images from training only
        # if drop_no_damage and self.is_train:
        #     before = len(self.files)
        #     self.files = [
        #         f for f in self.files
        #         if np.any(cv2.imread(str(self.mask_dir / f), cv2.IMREAD_GRAYSCALE) > 0)
        #     ]
        #     print(f"Dropped {before - len(self.files)} no-damage-only images")
        
        print(f"Loaded {split} set: {len(self.files)} samples")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        fname = self.files[idx]
        
        # load images
        pre  = cv2.cvtColor(cv2.imread(str(self.pre_dir  / fname)), cv2.COLOR_BGR2RGB)
        post = cv2.cvtColor(cv2.imread(str(self.post_dir / fname)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.mask_dir / fname), cv2.IMREAD_GRAYSCALE)
        
        # safety net for missing masks
        if mask is None:
            mask = np.zeros((pre.shape[0], pre.shape[1]), dtype=np.uint8)
        
        # apply augmentation on the fly during training
        # has_damage = np.any(mask > 0)
        pre, post, mask = augment(pre, post, mask, is_train=self.is_train)
        
        # normalize to [0,1] and convert to tensor
        pre  = torch.from_numpy(pre.copy().transpose(2, 0, 1)).float()  / 255.0
        post = torch.from_numpy(post.copy().transpose(2, 0, 1)).float() / 255.0
        mask = torch.from_numpy(mask.copy()).long()
        
        return pre, post, mask


# returns train, val and test dataloaders
def get_dataloaders(preprocessed_dir, batch_size=8, num_workers=4):
    
    train_dataset = xBDDataset(preprocessed_dir, split='train')
    val_dataset   = xBDDataset(preprocessed_dir, split='val')
    test_dataset  = xBDDataset(preprocessed_dir, split='test')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, 
                              shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, 
                              shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader