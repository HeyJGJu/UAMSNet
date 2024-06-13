# -*- codeing = utf-8 -*-
# @Author : linxihao
# @File : WORD_Loader.py
# @Software : PyCharm
import torch
import numpy as np
import cv2  
import random

from skimage.io import imread
from skimage import color

import torch
import torch.utils.data
from torchvision import datasets, models, transforms

def mask_to_onehot(mask, palette):
    """
    Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of class.
    """
    semantic_map = []
    for colour in palette:
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
    return semantic_map

class WORD_Loader(torch.utils.data.Dataset):

    def __init__(self, img_paths, transform=None):
        self.img_paths = glob.glob(img_paths)
        self.mask_paths = glob.glob(img_paths.replace('npyImages', 'npyMasks'))
        self.transform = transform
        self.palette = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14],[15], [16]]
        # self.palette = [[0], [1], [2], [3], [4], [5], [6], [7], [8]]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        npimage = np.load(img_path)
        npmask = np.load(mask_path)

        npimage = npimage.transpose((2, 0, 1))
        
        
        unique_nums = np.unique(npmask)  
        unique_nums = unique_nums.tolist()  
        unique_nums.sort() 
        labelindex = [1 if x[0] in unique_nums else 0 for x in self.palette]
        labelindex = np.array(labelindex)

        npmask = npmask.reshape(npmask.shape[0],npmask.shape[1],1)
        npmask = mask_to_onehot(npmask,self.palette)
        # npmask = npmask[:, :, 0:17]
        npmask = npmask.transpose((2, 0, 1))
        npmask = npmask.astype("float32")
        npimage = npimage.astype("float32")


        return npimage,npmask,labelindex
