# -*- codeing = utf-8 -*-
# @Author : linxihao
# @File : attentionunet_model.py
# @Software : PyCharm
import torch
from model.unet_model import SoftDiceLoss,UNet
from model.unet_DAC import DACUNet
from model.unet_DCN import DCNUNet
from model.unet_DSC import DSCUNet
from model.MyModel import MyModel
from utils.dataset import WORD_Loader
from torch.nn.modules.loss import _Loss
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler
from torch import optim
import torch.nn as nn
import torch
import math
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from utils import helper

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# log
writer = SummaryWriter("myModel")

palette = [[0], [15], [31], [47], [63], [79], [95], [111], [127], [143], [159], [175], [191], [207], [223], [239], [255]]

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if(val != 1.0):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count

def diceCoeffv2(pred, gt, eps=1e-5):
    r""" computational formula：
        dice = (2 * tp) / (2 * tp + fp + fn)
    """

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    tp = torch.sum(gt_flat * pred_flat, dim=1)
    fp = torch.sum(pred_flat, dim=1) - tp
    fn = torch.sum(gt_flat, dim=1) - tp
    score = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return score.sum() / N




def onehot_to_onehot(predict, palette):
    tmp_pred = onehot_to_mask(predict.squeeze().permute(1, 2, 0), palette)
    tmp_pred = mask_to_onehot(tmp_pred,palette)
    return tmp_pred
def mask_to_onehot(mask, palette):
    """
    Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of classes.
    """
    device = mask.device  
    semantic_map = []
    for colour in palette:
        colour_tensor = torch.tensor(colour, device=device)  
        equality = torch.eq(mask, colour_tensor)  
        class_map = torch.all(equality, dim=-1)  
        semantic_map.append(class_map)
    semantic_map = torch.stack(semantic_map, dim=-1).float()  
    return semantic_map

def onehot_to_mask(mask, palette):
    """
    Converts a mask (H, W, K) to (H, W, C)
    """
    x = torch.argmax(mask, axis=-1)
    colour_codes = torch.tensor(palette).to(mask.device)
    x = colour_codes[x]
    return x.to(torch.uint8)
def getIndex(predict):
    tmp_pred = onehot_to_mask(predict.squeeze().permute(1, 2, 0), palette)
    tmp_pred = tmp_pred.squeeze(2)
    unique_nums = torch.unique(tmp_pred)  
    unique_nums = unique_nums.tolist()  
    unique_nums.sort()  
    labelindex = [1 if x[0] in unique_nums else 0 for x in palette]
    return labelindex

def tversky(pred, gt, beta=0.7, weights=None):
    r""" computational formula：
        dice = (tp) / (tp + alpha*fp + beta*fn)
    """
    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)
    alpha = 1.0 - beta
    tp = torch.sum(gt_flat * pred_flat, dim=1)
    fp = torch.sum(pred_flat, dim=1) - tp
    fn = torch.sum(gt_flat, dim=1) - tp
    score = (tp) / (tp + alpha*fp + beta*fn)
    return score.sum() / N

class TverskyLoss(_Loss):

    def __init__(self, num_classes):
        super(TverskyLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, y_pred, y_true):
        class_tversky = []
        for i in range(1, self.num_classes):
            class_tversky.append(tversky(y_pred[:, i:i + 1, :], y_true[:, i:i + 1, :]))
        mean_tversky = sum(class_tversky) / len(class_tversky)
        return 1 - mean_tversky

class MyTverskyLoss(_Loss):

    def __init__(self,num_classes):
        super(MyTverskyLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, y_pred, y_true, labelindex):
        class_dice = []
        for i in range(len(labelindex)):
            for j in range(0,self.num_classes):
                if(labelindex[i][j] == 1):
                    class_dice.append(tversky(y_pred[i:i+1, j:j + 1, :], y_true[i:i+1, j:j + 1, :]))
        mean_dice = sum(class_dice) / len(class_dice)
        return 1 - mean_dice

def combinLoss(pred, gt)：
    loss1 = diceCoeffv2(pred,gt)
    loss2 = nn.BCEWithLogitsLoss()
    loss = loss1 + loss2(pred,gt)
    return loss
  
   
class MyLoss(_Loss):

    def __init__(self,num_classes):
        super(newMyDiceLoss, self).__init__()
        self.num_classes = num_classes
    def forward(self, y_pred, y_true, labelindex,epoch):
        my_list = []
        for i in range(len(y_pred)):
            list = getIndex(y_pred[i:i+1,:,:,:])
            my_list.append(list)
        class_dice = []
        
        for i in range(len(labelindex)):
            for j in range(0,self.num_classes):
                if(labelindex[i][j] == 1):
                    class_dice.append(combinLoss(y_pred[i:i + 1, j:j + 1, :], y_true[i:i + 1, j:j + 1, :]))
                if ((labelindex[i][j] != 1 and my_list[i][j] == 1)):
                    class_dice.append((1/1+math.exp(50-epoch)) * combinLoss(y_pred[i:i + 1, j:j + 1, :], y_true[i:i + 1, j:j + 1, :]))
        mean_dice = sum(class_dice) / len(class_dice)
        return 1 - mean_dice



def train_net(net, device, data_path,val_path, epochs=150, batch_size=1, lr=0.00001):
    log = pd.DataFrame(index=[], columns=[
        'epoch', 'lr', 'liver', 'spleen', 'left_kidney', 'right_kidney', 'stomach', 'gallbladder', 'esophagus',
        'pancreas', 'duodenum', 'colon',
        'intestine', 'adrenal', 'rectum', 'bladder', 'Head_of_femur_L', 'Head_of_femur_R', 'valmean'
    ])
    # load dateset
    word_dataset = WORD_Loader(data_path)
    val_dataset = WORD_Loader(val_path)

    per_epoch_num = len(word_dataset) / batch_size
    train_loader = torch.utils.data.DataLoader(dataset=word_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                               batch_size=1,
                                               shuffle=False)

    
    # optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    # loss function
    # criterion = nn.BCEWithLogitsLoss()
    criterion = SoftDiceLoss(17)
    criterion1 = nn.BCELoss()
    criterion2 = MyLoss(17)
    # best_score，
    best_score = 0
    best_loss = float('inf')
    # training
    with tqdm(total=epochs*per_epoch_num) as pbar:
        for epoch in range(epochs):
            
            net.train()
            train_losses = []
           
            for image, label,labelindex in train_loader:
                optimizer.zero_grad()
                
                image = image.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.float32)
               
                pred = net(image)
                pred = torch.sigmoid(pred)

                # calculate loss
               
                labelindex = labelindex.tolist()
                loss = criterion2(pred, label)
                # loss =criterion2(pred, label, labelindex,epoch)
                # dice = diceCoeff(pred, label)
                # if epoch > 10:
                #     if loss < best_loss:
                #         best_loss = loss
                #         torch.save(net.state_dict(), '/checkpoints/minloss_model.pth')
                train_losses.append(loss.item())
                
                class_dice = []
                for i in range(1, 17):
                    cur_dice = diceCoeffv2(pred[:, i:i + 1, :, :], label[:, i:i + 1, :, :])
                    # print(i, "------------", cur_dice.item())
                    class_dice.append(cur_dice.item())
               
                # print('epoch = ', epoch, 'Loss/train', loss.item())
                # print('{}/{}：Loss/train'.format(epoch + 1, epochs), loss.item())
                # Update parameter
                loss.backward()
                optimizer.step()
                pbar.update(1)
            train_loss = np.average(train_losses)
            
            writer.add_scalar('main_loss', train_loss, epoch)
            # valid model
            net.eval()
            losses = AverageMeter()
            ious = AverageMeter()
            dices_1s = AverageMeter()
            dices_2s = AverageMeter()
            dices_3s = AverageMeter()
            dices_4s = AverageMeter()
            dices_5s = AverageMeter()
            dices_6s = AverageMeter()
            dices_7s = AverageMeter()
            dices_8s = AverageMeter()
            dices_9s = AverageMeter()
            dices_10s = AverageMeter()
            dices_11s = AverageMeter()
            dices_12s = AverageMeter()
            dices_13s = AverageMeter()
            dices_14s = AverageMeter()
            dices_15s = AverageMeter()
            dices_16s = AverageMeter()
            for val_image, val_label, val_labelindex in val_loader:
                val_image = val_image.to(device=device, dtype=torch.float32)
                val_label = val_label.to(device=device, dtype=torch.float32)
                val_pred = net(val_image)
                val_pred = torch.sigmoid(val_pred)
                # val_pred = onehot_to_onehot(val_pred,palette)
                # val_pred = val_pred.permute(2, 0, 1).unsqueeze(0)

       
                class_dice = []
                for i in range(1, 17):
                    cur_dice = diceCoeffv2(val_pred[:, i:i + 1, :, :], val_label[:, i:i + 1, :, :])
                    # print(i, "------------", cur_dice.item())
                    class_dice.append(cur_dice.item())
                dices_1s.update(class_dice[0], val_pred.shape[0])
                dices_2s.update(class_dice[1], val_pred.shape[0])
                dices_3s.update(class_dice[2], val_pred.shape[0])
                dices_4s.update(class_dice[3], val_pred.shape[0])
                dices_5s.update(class_dice[4], val_pred.shape[0])
                dices_6s.update(class_dice[5], val_pred.shape[0])
                dices_7s.update(class_dice[6], val_pred.shape[0])
                dices_8s.update(class_dice[7], val_pred.shape[0])
                dices_9s.update(class_dice[8], val_pred.shape[0])
                dices_10s.update(class_dice[9], val_pred.shape[0])
                dices_11s.update(class_dice[10], val_pred.shape[0])
                dices_12s.update(class_dice[11], val_pred.shape[0])
                dices_13s.update(class_dice[12], val_pred.shape[0])
                dices_14s.update(class_dice[13], val_pred.shape[0])
                dices_15s.update(class_dice[14], val_pred.shape[0])
                dices_16s.update(class_dice[15], val_pred.shape[0])

            all_avg = (dices_1s.avg + dices_2s.avg + dices_3s.avg + dices_4s.avg + dices_5s.avg + dices_6s.avg + dices_7s.avg + dices_8s.avg + dices_9s.avg + dices_10s.avg + dices_11s.avg + dices_12s.avg + dices_13s.avg + dices_14s.avg + dices_15s.avg + dices_16s.avg) / 16
            
            writer.add_scalar('val_AVG_dice', all_avg, epoch)
            tmp = pd.Series([
                epoch,
                lr,
                dices_1s.avg, dices_2s.avg, dices_3s.avg, dices_4s.avg, dices_5s.avg, dices_6s.avg, dices_7s.avg,
                dices_8s.avg, dices_9s.avg, dices_10s.avg, dices_11s.avg, dices_12s.avg, dices_13s.avg, dices_14s.avg,
                dices_15s.avg, dices_16s.avg,all_avg,
            ], index=[
                'epoch', 'lr','liver', 'spleen', 'left_kidney', 'right_kidney', 'stomach', 'gallbladder',
                'esophagus', 'pancreas', 'duodenum', 'colon',
                'intestine', 'adrenal', 'rectum', 'bladder', 'Head_of_femur_L', 'Head_of_femur_R', 'valmean'
            ])
            log = log.append(tmp, ignore_index=True)
            log.to_csv('checkpoints/log.csv', index=False)
            if all_avg >= best_score:
                best_score = all_avg
                torch.save(net.state_dict(),'/checkpoints/epoch{}-{:.4f}.pth'.format(epoch,all_avg))
        torch.save(net.state_dict(), '/checkpoints/last_model.pth')


if __name__ == "__main__":
    device = torch.device('cuda:0')
    net = MyModel(n_channels=3, n_classes=17)  # todo edit input_channels n_classes
    # net = DCNUNetplus(n_channels=3, n_classes=17)  # todo edit input_channels n_classes
    net.to(device=device)
    data_path = "/dataset/Train" 
    val_path = "dataset/Val"
    train_net(net, device, data_path, val_path, epochs=100, batch_size=4)
    

