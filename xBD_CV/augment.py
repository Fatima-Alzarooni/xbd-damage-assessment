# augmentation functions 
# applied on during training to pre, post, and mask triples
# geometric transforms are applied to all three
# color transforms are applied to pre and post only, NOT mask

import cv2
import numpy as np
import random


def flip(pre, post, mask, flip_code):
    # flip_code: 1 = horizontal, 0 = vertical
    return (
        cv2.flip(pre,  flip_code),
        cv2.flip(post, flip_code),
        cv2.flip(mask, flip_code)
    )


def rotate(pre, post, mask, angle):
    # angle: 90, 180, or 270
    k = angle // 90
    return (
        np.rot90(pre,  k).copy(),
        np.rot90(post, k).copy(),
        np.rot90(mask, k).copy()
    )

def brightness_contrast(pre, post, mask, alpha, beta):
    # alpha: contrast (>1 = more contrast, <1 = less)
    # beta:  brightness (>0 = brighter, <0 = darker)
    return (
        cv2.convertScaleAbs(pre,  alpha=alpha, beta=beta),
        cv2.convertScaleAbs(post, alpha=alpha, beta=beta),
        mask
    )

def gaussian_noise(pre, post, mask, sigma=15):
    # adds random sensor noise to images only
    noise = np.random.normal(0, sigma, pre.shape).astype(np.float32)
    pre_n  = np.clip(pre.astype(np.float32)  + noise, 0, 255).astype(np.uint8)
    post_n = np.clip(post.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return pre_n, post_n, mask

def random_crop(pre, post, mask, crop_size=192):
    # crops a random region then resizes back to original size, 265
    h, w = pre.shape[:2]
    x = np.random.randint(0, w - crop_size)
    y = np.random.randint(0, h - crop_size)
    
    pre_c  = cv2.resize(pre[y:y+crop_size,  x:x+crop_size], (w, h))
    post_c = cv2.resize(post[y:y+crop_size, x:x+crop_size], (w, h))
    mask_c = cv2.resize(mask[y:y+crop_size, x:x+crop_size], (w, h),
                        interpolation=cv2.INTER_NEAREST)
    return pre_c, post_c, mask_c

#  main augment function called during training                        
def augment(pre, post, mask, is_train=True):

    # randomly applies augmentations to a pre/post/mask triple
    # only called during training (is_train=True)
    # returns augmented pre, post, mask as numpy arrays

    if not is_train:
        return pre, post, mask
    
    # geometric transforms - applied to all three
    # horizontal flip with 50% chance
    if random.random() < 0.5:
        pre, post, mask = flip(pre, post, mask, flip_code=1)
    
    # vertical flip with 50% chance
    if random.random() < 0.5:
        pre, post, mask = flip(pre, post, mask, flip_code=0)
    
    # random rotation with 50% chance
    if random.random() < 0.5:
        angle = random.choice([90, 180, 270])
        pre, post, mask = rotate(pre, post, mask, angle)
    
    # random crop with 50% chance
    if random.random() < 0.5:
        pre, post, mask = random_crop(pre, post, mask, crop_size=192)
    
    # color transforms - applied to images only
    # brightness/contrast with 40% chance
    if random.random() < 0.4:
        alpha = random.uniform(0.8, 1.2)
        beta  = random.randint(-30, 30)
        pre, post, mask = brightness_contrast(pre, post, mask, alpha, beta)
    
    # gaussian noise with 30% chance
    if random.random() < 0.3:
        sigma = random.uniform(10, 25)
        pre, post, mask = gaussian_noise(pre, post, mask, sigma)
    
    return pre, post, mask