import os
import sys
import gc
import ast
import cv2
import time
import timm
import pickle
import random
import argparse
import warnings
import numpy as np
import pandas as pd
from glob import glob
import nibabel as nib
from PIL import Image
from tqdm import tqdm
import albumentations
from albumentations.pytorch import ToTensorV2
from pylab import rcParams
import matplotlib.pyplot as plt
# import segmentation_models_pytorch as smp
from sklearn.model_selection import KFold, StratifiedKFold
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from monai.transforms import Resize
import  monai.transforms as transforms


# rcParams['figure.figsize'] = 20, 8
# device = torch.device('cuda')
# torch.backends.cudnn.benchmark = True
import sys
###############################################################
#segmentation nii shape: 512 512 num
image_sizes = [128, 128, 128]
seg_sizes = [128, 128, 128]
crop_sizes = [224,224]

n_slice = 15
in_chans = 6
neighbours = int((in_chans - 2) / 2)

R = Resize(image_sizes)

init_lr = 3e-3
batch_size = 4
drop_rate = 0.
drop_path_rate = 0.
loss_weights = [1, 1]
p_mixup = 0.1


use_amp = True
num_workers = 4
out_dim = 5

n_epochs = 1000

log_dir = './logs'
model_dir = './models'
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)


transforms_train = transforms.Compose([
    transforms.RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=1),
    transforms.RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=2),
    transforms.RandAffined(keys=["image", "mask"], translate_range=[int(x*y) for x, y in zip(image_sizes, [0.3, 0.3, 0.3])], padding_mode='zeros', prob=0.7),
    transforms.RandGridDistortiond(keys=("image", "mask"), prob=0.5, distort_limit=(-0.01, 0.01), mode="nearest"),
])

transforms_valid = transforms.Compose([
])
kernel_type = 'timm3d_res18d_unet4b_128_128_128_dsv2_flip12_shift333p7_gd1p5_bs4_lr3e4_20x50ep'
load_kernel = None
load_last = True
n_blocks = 4
n_folds = 5
backbone = 'resnet18d'
DEBUG = False
def load_png(path):
    img = Image.open(path)
    data = np.array(img)
    data = cv2.resize(data, (image_sizes[0], image_sizes[1]), interpolation = cv2.INTER_LINEAR)
    return data


def load_png_line_par(path):
    t_paths = sorted(glob(path+"_*.png"),
                     key=lambda x: int(x.split('/')[-1].split(".")[0]))

    n_scans = len(t_paths)
    indices = np.quantile(list(range(n_scans)), np.linspace(0., 1., image_sizes[2])).round().astype(int)
    t_paths = [t_paths[i] for i in indices]

    images = []
    for filename in t_paths:
        images.append(load_png(filename))
    images = np.stack(images, -1)

    images = images - np.min(images)
    images = images / (np.max(images) + 1e-4)
    images = (images * 255).astype(np.uint8)
    return images


def load_sample(row, has_mask=True):
    image = load_png_line_par(row.image_path)
    if image.ndim < 4:
        image = np.expand_dims(image, 0).repeat(3, 0)  # to 3ch

    if has_mask:
        mask_org = nib.load(row.mask_file).get_fdata()
        shape = mask_org.shape
        mask_org = mask_org.transpose(1, 0, 2)[::-1, :, :]  # (d, w, h)
        mask = np.zeros((out_dim, shape[1], shape[0], shape[2]))
        for cid in range(out_dim):
            mask[cid] = (mask_org == (cid + 1))
        mask = mask.astype(np.uint8) * 255
        mask = R(mask).numpy()

        return image, mask
    else:
        return image


class SEGDataset(Dataset):
    def __init__(self, df, mode, transform):
        self.df = df.reset_index()
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]

        ### using local cache
        #         image_file = os.path.join(data_dir, f'{row.StudyInstanceUID}.npy')
        #         mask_file = os.path.join(data_dir, f'{row.StudyInstanceUID}_mask.npy')
        #         image = np.load(image_file).astype(np.float32)
        #         mask = np.load(mask_file).astype(np.float32)
        if self.mode == 'test':
            image =  load_sample(row, has_mask=False)
            res = self.transform({'image': image})
            image = res['image'] / 255.
            image = torch.tensor(image).float()
            return image, row['series_id']
        else:
            image, mask = load_sample(row, has_mask=True)

            res = self.transform({'image': image, 'mask': mask})
            image = res['image'] / 255.
            mask = res['mask']
            mask = (mask > 127).astype(np.float32)

            image, mask = torch.tensor(image).float(), torch.tensor(mask).float()

            return image, mask

class TimmSegModel(nn.Module):
    def __init__(self, backbone, segtype='unet', pretrained=False):
        super(TimmSegModel, self).__init__()

        self.encoder = timm.create_model(
            backbone,
            in_chans=3,
            features_only=True,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            pretrained=pretrained
        )
        g = self.encoder(torch.rand(1, 3, 64, 64))
        encoder_channels = [1] + [_.shape[1] for _ in g]
        decoder_channels = [256, 128, 64, 32, 16]
        if segtype == 'unet':
            self.decoder = smp.unet.decoder.UnetDecoder(
                encoder_channels=encoder_channels[:n_blocks+1],
                decoder_channels=decoder_channels[:n_blocks],
                n_blocks=n_blocks,
            )

        self.segmentation_head = nn.Conv2d(decoder_channels[n_blocks-1], out_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self,x):
        global_features = [0] + self.encoder(x)[:n_blocks]
        seg_features = self.decoder(*global_features)
        seg_features = self.segmentation_head(seg_features)
        return seg_features


from timm.models.layers.conv2d_same import Conv2dSame
from conv3d_same import Conv3dSame


def convert_3d(module):
    module_output = module
    if isinstance(module, torch.nn.BatchNorm2d):
        module_output = torch.nn.BatchNorm3d(
            module.num_features,
            module.eps,
            module.momentum,
            module.affine,
            module.track_running_stats,
        )
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig

    elif isinstance(module, Conv2dSame):
        module_output = Conv3dSame(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size[0],
            stride=module.stride[0],
            padding=module.padding[0],
            dilation=module.dilation[0],
            groups=module.groups,
            bias=module.bias is not None,
        )
        module_output.weight = torch.nn.Parameter(module.weight.unsqueeze(-1).repeat(1, 1, 1, 1, module.kernel_size[0]))

    elif isinstance(module, torch.nn.Conv2d):
        module_output = torch.nn.Conv3d(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size[0],
            stride=module.stride[0],
            padding=module.padding[0],
            dilation=module.dilation[0],
            groups=module.groups,
            bias=module.bias is not None,
            padding_mode=module.padding_mode
        )
        module_output.weight = torch.nn.Parameter(module.weight.unsqueeze(-1).repeat(1, 1, 1, 1, module.kernel_size[0]))

    elif isinstance(module, torch.nn.MaxPool2d):
        module_output = torch.nn.MaxPool3d(
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            ceil_mode=module.ceil_mode,
        )
    elif isinstance(module, torch.nn.AvgPool2d):
        module_output = torch.nn.AvgPool3d(
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            ceil_mode=module.ceil_mode,
        )

    for name, child in module.named_children():
        module_output.add_module(
            name, convert_3d(child)
        )
    del module

    return module_output

from typing import Any, Dict, Optional
def binary_dice_score(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    threshold: Optional[float] = None,
    nan_score_on_empty=False,
    eps: float = 1e-7,
) -> float:

    if threshold is not None:
        y_pred = (y_pred > threshold).to(y_true.dtype)

    intersection = torch.sum(y_pred * y_true).item()
    cardinality = (torch.sum(y_pred) + torch.sum(y_true)).item()

    score = (2.0 * intersection) / (cardinality + eps)

    has_targets = torch.sum(y_true) > 0
    has_predicted = torch.sum(y_pred) > 0

    if not has_targets:
        if nan_score_on_empty:
            score = np.nan
        else:
            score = float(not has_predicted)
    return score


def multilabel_dice_score(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    threshold=None,
    eps=1e-7,
    nan_score_on_empty=False,
):
    ious = []
    num_classes = y_pred.size(0)
    for class_index in range(num_classes):
        iou = binary_dice_score(
            y_pred=y_pred[class_index],
            y_true=y_true[class_index],
            threshold=threshold,
            nan_score_on_empty=nan_score_on_empty,
            eps=eps,
        )
        ious.append(iou)

    return ious


def dice_loss(input, target):
    input = torch.sigmoid(input)
    smooth = 1.0
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return 1 - ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))


def bce_dice(input, target, loss_weights=loss_weights):
    loss1 = loss_weights[0] * nn.BCEWithLogitsLoss()(input, target)
    loss2 = loss_weights[1] * dice_loss(input, target)
    return (loss1 + loss2) / sum(loss_weights)




def mixup(input, truth, clip=[0, 1]):
    indices = torch.randperm(input.size(0))
    shuffled_input = input[indices]
    shuffled_labels = truth[indices]

    lam = np.random.uniform(clip[0], clip[1])
    input = input * lam + shuffled_input * (1 - lam)
    return input, truth, shuffled_labels, lam


reduce_rate = 1
transforms_train_ex = albumentations.Compose([
    # albumentations.Resize(512, 512, p=1),
    albumentations.LongestMaxSize(512),
    albumentations.PadIfNeeded(512, 512, border_mode=0, p=1),
    albumentations.Perspective(p=0.5*reduce_rate),
    albumentations.VerticalFlip(p=0.5*reduce_rate),
    albumentations.HorizontalFlip(p=0.5*reduce_rate),
    albumentations.RandomContrast(p=0.75*reduce_rate),
    albumentations.Rotate(p=0.5*reduce_rate, limit=(45, -45)),
    albumentations.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, border_mode=cv2.BORDER_CONSTANT, p=0.75*reduce_rate),
    albumentations.OneOf([
        albumentations.MotionBlur(blur_limit=3),
        albumentations.MedianBlur(blur_limit=3),
        albumentations.GaussianBlur(blur_limit=3),
        albumentations.GaussNoise(var_limit=(3.0, 9.0)),
    ], p=0.5),
    albumentations.OneOf([
        albumentations.OpticalDistortion(distort_limit=1.),
        albumentations.GridDistortion(num_steps=5, distort_limit=1.),
    ], p=0.5),

    albumentations.Cutout(max_h_size=int(512 * 0.1), max_w_size=int(512 * 0.1), num_holes=6, p=0.75*reduce_rate, fill_value=0, always_apply=True),
    ToTensorV2()], bbox_params=albumentations.BboxParams(format='albumentations', label_fields=['class_labels'])
)

transforms_valid_ex = albumentations.Compose([
    # albumentations.Resize(512, 512, p=1),
    albumentations.LongestMaxSize(512),
    albumentations.PadIfNeeded(512, 512, border_mode=0, p=1),
    ToTensorV2()], bbox_params=albumentations.BboxParams(format='albumentations', label_fields=['class_labels'])
)
# cfg.shift_limit = 0.2
# cfg.scale_limit = 0.2
# cfg.rotate_limit = 20
# cfg.RandomContrast = 0.2
# cfg.hflip = 0.5
# cfg.transpose = 0.
# cfg.vflip = 0.
# cfg.holes = 0
# cfg.hole_size = 0.1
# cfg.norm_mean = [0.22363983, 0.18190407, 0.2523437]
# cfg.norm_std = [0.32451536, 0.2956294,  0.31335256]


class SegNetEx(nn.Module):
    def __init__(self, pretrained,backbone='efficientnet_b1'):
        super(SegNetEx, self).__init__()

        self.offline_inference = not pretrained
        self.backbone = timm.create_model(backbone,
                                          pretrained=pretrained,
                                          in_chans=1 * 3,
                                          num_classes=4)
        self.bboxfc = self.backbone.classifier
        self.backbone.classifier = torch.nn.Identity()
        self.fc = torch.nn.Linear(in_features=self.bboxfc.in_features,
                                  out_features=1,
                                  bias=True)
        self.criterion = torch.nn.L1Loss()
        self.criterionbce = nn.BCEWithLogitsLoss()

    def forward(self, batch):
        x = batch['image']
        x = self.backbone(x)
        logits_cls = self.fc(x)
        logits_bb = self.bboxfc(x)

        out = torch.cat((logits_bb, torch.sigmoid(logits_cls)), 1)

        if not self.offline_inference:
            labels_bb = batch['labels'][:, :4]
            labels_cls = batch['labels'][:, -1:].clip(0.05, 0.95)
            idx = labels_cls.flatten() > 0.5
            if labels_bb[idx].shape[0] == 0:
                loss_cls = self.criterionbce(logits_cls, labels_cls)
                loss = 1 * loss_cls
                loss_bb = - torch.tensor([0.])
            else:
                loss_bb = self.criterion(logits_bb[idx], labels_bb[idx])
                loss_cls = self.criterionbce(logits_cls, labels_cls)
                loss = 0.8 * loss_bb + 0.2 * loss_cls
                if loss_bb != loss_bb:
                    print(logits_bb[idx])
                    print(labels_bb[idx])
        else:
            loss = -1
            loss_bb = -1
            loss_cls = -1

        return {'preds': out,
                'loss': loss,
                'loss_bbox': loss_bb,
                'loss_cls': loss_cls}


class SEGDatasetEx(Dataset):
    def __init__(self, df, mode, transform):
        self.df = df
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index, :]
        if self.mode == 'test':
            t_paths = sorted(glob(row['image_path'] + "*.npy"), key=lambda x: int(x.split('/')[-1].split(".")[0]))
            images = []
            for filename in t_paths:
                images.append(np.load(filename))
            images = np.stack(images, -1)
            images = images - np.min(images)
            images = images / (np.max(images) + 1e-4)
            images = (images * 255).astype(np.uint8)
            try:
                transformed = self.transform(image=images, bboxes=np.array([[0, 0, 1, 1]]), class_labels=['has_bbox'])
            except:
                transformed = self.transform(image=images, bboxes=np.array([[0, 0, 1, 1]]), class_labels=['has_bbox'])
                print(row)
            images = transformed['image'] / 255.
            images = np.stack([images[:-2, :, :], images[1:-1, :, :], images[2:, :, :]], axis=1)  # [C-2, 3, H, W]

            return images.astype(np.float32), row['patient_id'], row['series_id']
        else:
            if row['extravasation_injury']:
                if row['flag']:
                    idx = random.choice(range(len(row['image_path'])))
                    bbox = row[['x1', 'y1', 'x2', 'y2']].tolist()
                    bbox = np.array(bbox)[:, idx] / 512
                    bbox = bbox[np.newaxis, :]
                    path = '/disk1/tanxin/train_data_npy/' + str(row['patient_id']) + '_' + str(row['series_id'][idx]) + '_'
                    up_path = path + str(row['png_number'][idx] - 1).zfill(4) + '.npy'
                    down_path = path + str(row['png_number'][idx] + 1).zfill(4) + '.npy'
                    images = [np.load(up_path), np.load(row['image_path'][idx]), np.load(down_path)]
                else:
                    idx = random.choice(range(len(row['image_path'])))
                    bbox = row[['x1', 'y1']].tolist() + row[['x2', 'y2']].tolist()
                    bbox = np.array([bbox]) / 512
                    path = '/disk1/tanxin/train_data_npy/' + str(row['patient_id']) + '_' + str(row['series_id'][idx]) + '_' + '*.npy'
                    image_files = sorted(glob(path), key=lambda x: int(x.split('/')[-1].split(".")[0]))
                    indices = [index for index, element in enumerate(row['series_id']) if element == row['series_id'][idx]]
                    while True:
                        idx = random.choice(list(set(range(1, len(image_files)-1)) - set(row['png_number'][i] for i in indices)))
                        try:
                            images = [np.load(image_files[idx - 1]), np.load(image_files[idx]), np.load(image_files[idx + 1])]
                            break
                        except:
                            print(image_files[idx - 1], image_files[idx], image_files[idx + 1])
            else:
                idx = random.choice(range(len(row['image_path'])))
                bbox = row[['x1', 'y1']].tolist() + row[['x2', 'y2']].tolist()
                bbox = np.array([bbox]) / 512
                image_files = sorted(glob(row['image_path'][idx]), key=lambda x: int(x.split('/')[-1].split(".")[0]))
                idx = random.choice(range(1, len(image_files)-1))
                images = [np.load(image_files[idx-1]), np.load(image_files[idx]), np.load(image_files[idx+1])]
            images = np.stack(images, -1)
            images = images - np.min(images)
            images = images / (np.max(images) + 1e-4)
            images = (images * 255).astype(np.uint8)
            try:
                transformed = self.transform(image=images, bboxes=bbox, class_labels=['has_bbox'])
            except:
                # transformed = self.transform(image=images, bboxes=bbox, class_labels=['has_bbox'])
                # print(images.shape)
                # print(row)
                print(bbox)
                exit(0)
            images = transformed['image']
            labels = transformed['bboxes']
            if len(labels) == 0:
                labels = [0.0, 0.0, 1.0, 1.0]
                has_bbox = 0
            else:
                labels = list(labels[0])
                has_bbox = row['has_bbox']
            images = images / 255.
            labels = torch.tensor(labels + [has_bbox])
            if labels.shape[0] == 0:
                print(row)
                print(labels)
                exit(0)
            return {'image': images.float(), 'labels': labels}

# if __name__ == '__main__':
#     np.load('/disk1/tanxin/train_data_npy/31474_20619_0105.npy')
#     np.load('/disk1/tanxin/train_data_npy/31474_20619_0106.npy')
#     np.load('/disk1/tanxin/train_data_npy/31474_20619_0107.npy')