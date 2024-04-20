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
from PIL import Image
from tqdm import tqdm
import albumentations
from pylab import rcParams
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold
import sklearn.metrics
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
DEBUG = False
rcParams['figure.figsize'] = 20, 8
device = torch.device('cuda')
torch.backends.cudnn.benchmark = True

kernel_type = 'crpo2_convnextbase_224_15_6ch_augv2_mixupp5_dr3_rov1p2_bs16_lr50e6_eta23e7_wd01'
load_kernel = None
load_last = True

n_folds = 5
backbone = 'convnext_base_in22ft1k'

image_size = 224
n_slice = 15
in_chans = 6
neighbours = int((in_chans - 2) / 2)

init_lr = 200e-7
eta_min = 92e-8
# init_lr = 1e-5
# eta_min = 1e-6
weight_decay = 0.1
batch_size = 8
drop_rate = 0.
drop_rate_last = 0.3
drop_path_rate = 0.
p_mixup = 0.5
p_rand_order_v1 = 0.2

data_dir = '/home/tanxin/Kaggle'
use_amp = True
num_workers = 30
out_dim = 1

n_epochs = 100

log_dir = './logs'
model_dir = './models'
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda:0')

transforms_train = albumentations.Compose([
    albumentations.Resize(image_size, image_size),
    albumentations.HorizontalFlip(p=0.5),
    albumentations.VerticalFlip(p=0.5),
    albumentations.Transpose(p=0.5),
    albumentations.RandomBrightness(limit=0.1, p=0.7),
    albumentations.ShiftScaleRotate(shift_limit=0.3, scale_limit=0.3, rotate_limit=45, border_mode=4, p=0.7),

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

    albumentations.Cutout(max_h_size=int(image_size * 0.5), max_w_size=int(image_size * 0.5), num_holes=1, p=0.5),
])

transforms_valid = albumentations.Compose([
    albumentations.Resize(image_size, image_size),
])


class CLSDataset(Dataset):
    def __init__(self, df, mode, transform, target):
        self.df = df
        self.mode = mode
        self.transform = transform
        self.target = target

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]

        images = []

        image = np.load(f'/home/tanxin/Kaggle/crop_{self.target}/{int(row.series_id)}_{self.target}.npy')
        ran = n_slice

        #print(image.shape)
        image, mask = image[:, :, :image.shape[2]-ran], image[:, :, image.shape[2]-ran:] * 255

        img_indices = np.quantile(list(range(neighbours,image.shape[2]-neighbours)), np.linspace(0., 1., n_slice)).round().astype(int)
        mask_indices = [math.floor(i/image.shape[2]*ran) for i in img_indices]
        #mask_indices = np.quantile(list(range(ran)), np.linspace(neighbours/image.shape[2],1-neighbours/image.shape[2],n_slice))
        #print(image.shape[2],mask_indices,ran,img_indices)
        for ind in range(len(mask_indices)):

            image_slice = image[:, :, img_indices[ind]-neighbours:img_indices[ind]+neighbours+1]

            mask_slice = mask[:, :, mask_indices[ind]][:, :, np.newaxis]
            image_slice = np.concatenate([image_slice, mask_slice], axis=2).astype(np.uint8)
            try:
                image_slice = self.transform(image=image_slice)['image']
            except:
                print(f'{int(row.series_id)}_{self.target}.npy')
                exit(0)
            image_slice = image_slice.transpose(2, 0, 1).astype(np.float32) / 255.
            images.append(image_slice)

        images = np.stack(images, 0)
        #print(images.shape)
        if self.mode != 'test':
            images = torch.tensor(images).float()
            labels = torch.tensor(row[self.target]).to(torch.long)
            if self.mode == 'train' and random.random() < p_rand_order_v1:
                indices = torch.randperm(images.size(0))
                images = images[indices]
            return images, labels
        else:
            return torch.tensor(images).float()


class TimmModel(nn.Module):
    def __init__(self, backbone, pretrained=False):
        super(TimmModel, self).__init__()

        self.encoder = timm.create_model(
            backbone,
            in_chans=in_chans,
            num_classes=out_dim,
            features_only=False,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            pretrained=pretrained
        )

        if 'efficient' in backbone:
            hdim = self.encoder.conv_head.out_channels
            self.encoder.classifier = nn.Identity()
        elif 'convnext' in backbone:
            hdim = self.encoder.head.fc.in_features
            self.encoder.head.fc = nn.Identity()


        self.lstm = nn.LSTM(hdim, 256, num_layers=2, dropout=drop_rate, bidirectional=True, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(drop_rate_last),
            nn.LeakyReLU(0.1),
        )
        self.head_last = nn.Sequential(
            nn.Linear(64 * n_slice, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(drop_rate_last),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 2),
        )

    def forward(self, x):  # (bs, nslice, ch, sz, sz)
        bs = x.shape[0]
        x = x.view(bs * n_slice, in_chans, image_size, image_size)
        feat = self.encoder(x)
        feat = feat.view(bs, n_slice, -1)
        feat, _ = self.lstm(feat)
        feat = feat.contiguous().view(bs * n_slice, -1)
        feat = self.head(feat)
        feat = feat.contiguous().view(bs, -1)
        feat = self.head_last(feat)

        return feat


def mixup(input, truth, clip=[0, 1]):
    indices = torch.randperm(input.size(0))
    shuffled_input = input[indices]
    shuffled_labels = truth[indices]

    lam = np.random.uniform(clip[0], clip[1])
    input = input * lam + shuffled_input * (1 - lam)
    return input, truth, shuffled_labels, lam

CE = nn.CrossEntropyLoss(weight=torch.tensor([1., 2,]).to(device))
# CE = nn.CrossEntropyLoss(weight=torch.tensor([1., 6]).to(device))
SFTMX = nn.Softmax(dim=1)

def criterion(logits, targets):
    loss = CE(logits, targets)
    return loss

# def score(submition, solution):
#     y_one_hot = F.one_hot(solution, num_classes=3)
#     y_true = y_one_hot / torch.sum(y_one_hot, dim=1).unsqueeze(1)
#     y_pred = SFTMX(submition)
#     y_pred = y_pred / torch.sum(y_pred, dim=1).unsqueeze(1)
#     score = sklearn.metrics.log_loss(
#         y_true=y_true.cpu().numpy(),
#         y_pred=y_pred.cpu().numpy(),
#         sample_weight=torch.sum(y_one_hot*torch.tensor([[1., 2, 4]]).to(device), dim=1).cpu().numpy()
#     )
#     return score

def score(submition, solution):
    y_one_hot = F.one_hot(solution, num_classes=3)
    y_true = y_one_hot / torch.sum(y_one_hot, dim=1).unsqueeze(1)
    y_pred = SFTMX(submition)
    y_pred = y_pred / torch.sum(y_pred, dim=1).unsqueeze(1)
    #print(y_true.shape,y_pred.shape)
    score = sklearn.metrics.log_loss(
        y_true=y_true.cpu().numpy(),
        y_pred=y_pred.cpu().numpy(),
        sample_weight=torch.sum(y_one_hot*torch.tensor([[1.,2.]]).to(device), dim=1).cpu().numpy()
    )
    return score


def train_func(model, loader_train, optimizer, scaler=None):
    model.train()
    train_loss = []
    bar = tqdm(loader_train)
    for images, targets in bar:
        optimizer.zero_grad()
        images = images.cuda()
        targets = targets.cuda()

        do_mixup = False
        if random.random() < p_mixup:
            do_mixup = True
            images, targets, targets_mix, lam = mixup(images, targets)

        with amp.autocast():
            logits = model(images)

            loss = criterion(logits, targets)
            if do_mixup:
                loss11 = criterion(logits, targets_mix)
                loss = loss * lam + loss11 * (1 - lam)

        train_loss.append(loss.item())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bar.set_description(f'smth:{np.mean(train_loss[-10:]):.4f}')

    return np.mean(train_loss)


def valid_func(model, loader_valid):
    model.eval()
    valid_loss = []
    gts = []
    outputs = []
    bar = tqdm(loader_valid)
    with torch.no_grad():
        for images, targets in bar:
            images = images.cuda()
            targets = targets.cuda()

            logits = model(images)
            loss = criterion(logits, targets)

            gts.append(targets)
            outputs.append(logits)
            valid_loss.append(loss.item())

            bar.set_description(f'smth:{np.mean(valid_loss[-10:]):.4f}')

        outputs = torch.cat(outputs)
        gts = torch.cat(gts)

        valid_loss = criterion(outputs, gts).item()

        #metric = score(outputs, gts)
        metric = roc_auc_score(gts.cpu(),outputs.softmax(dim=-1)[:,-1].cpu())
    return valid_loss, metric

def run(fold, df, target):

    log_file = os.path.join(log_dir, f'{kernel_type}_{target}.txt')
    model_file = os.path.join(model_dir, f'{kernel_type}_fold{fold}_{target}_best.pth')

    train_ = df[df['fold'] != fold].reset_index(drop=True)
    valid_ = df[df['fold'] == fold].reset_index(drop=True)
    dataset_train = CLSDataset(train_, 'train', transform=transforms_train, target=target)
    dataset_valid = CLSDataset(valid_, 'valid', transform=transforms_valid, target=target)

    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    loader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = TimmModel(backbone, pretrained=True)
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=init_lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    metric_best = -np.inf
    loss_min = np.inf
    best_epoch = 0
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, n_epochs, eta_min=eta_min)

    print(len(dataset_train), len(dataset_valid))

    for epoch in range(1, n_epochs+1):
        if epoch-best_epoch>20:
            return
        scheduler_cosine.step(epoch-1)

        print(time.ctime(), 'Epoch:', epoch)

        train_loss = train_func(model, loader_train, optimizer, scaler)
        valid_loss, metric = valid_func(model, loader_valid)

        content = time.ctime() + ' ' + f'Fold {fold}, Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {train_loss:.5f}, valid loss: {valid_loss:.5f}, metric: {(metric):.6f}.'
        print(content)
        with open(log_file, 'a') as appender:
            appender.write(content + '\n')

        if metric > metric_best:
            best_epoch = epoch
            print(f'metric_best ({metric_best:.6f} --> {metric:.6f}). Saving model ...')
#             if not DEBUG:
            torch.save(model.state_dict(), model_file)
            metric_best = metric

        # Save Last
        if not DEBUG:
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict() if scaler else None,
                    'score_best': metric_best,
                },
                model_file.replace('_best', '_last')
            )

    del model
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == '__main__':
    # dataset = CLSDataset(train, 'train', transform=transforms_train, target='liver')
    # loader = torch.utils.data.DataLoader(dataset_show, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    train = pd.read_csv("train.csv")
    train['bowel'] = (train.iloc[:, 1:3] == 1).idxmax(1)
    train['extravasation'] = (train.iloc[:, 3:5] == 1).idxmax(1)
    train['kidney'] = (train.iloc[:, 5:8] == 1).idxmax(1)
    train['liver'] = (train.iloc[:, 8:11] == 1).idxmax(1)
    train['spleen'] = (train.iloc[:, 11:14] == 1).idxmax(1)
    train = train.drop(
        columns=['bowel_healthy', 'bowel_injury', 'extravasation_healthy', 'extravasation_injury', 'kidney_healthy',
                 'kidney_low', 'kidney_high', 'liver_healthy', 'liver_low', 'liver_high',
                 'spleen_healthy', 'spleen_low', 'spleen_high'])
    train['bowel'] = train['bowel'].replace(['bowel_healthy', 'bowel_injury'], [0, 1])
    train['extravasation'] = train['extravasation'].replace(['extravasation_healthy', 'extravasation_injury'], [0, 1])
    train['kidney'] = train['kidney'].replace(['kidney_healthy', 'kidney_low', 'kidney_high'], [0, 1, 2])
    train['liver'] = train['liver'].replace(['liver_healthy', 'liver_low', 'liver_high'], [0, 1, 2])
    train['spleen'] = train['spleen'].replace(['spleen_healthy', 'spleen_low', 'spleen_high'], [0, 1, 2])
    train_series_meta = pd.read_csv("crop_df.csv")
    train_series_meta['aortic_hu'] = train_series_meta['aortic_hu'].abs()
    train = pd.merge(train, train_series_meta, on='patient_id')
    # ['patient_id', 'any_injury', 'bowel', 'extravasation', 'kidney', 'liver', 'spleen', 'series_id', 'aortic_hu', 'incomplete_organ']
    # train = train.loc[list(range(50))]

    # print(train[train['series_id']==-90.0].iloc[:, :-2])
    target = "bowel"
    train = train[train['0']!=0].reset_index()

    train['fold'] = 0
    skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    fold_size = len(train) // 5
    for i, (_,valid_ind) in enumerate(skf.split(train,train[target])):
        #print(valid_ind)
        train.loc[valid_ind, 'fold'] = i

    # run(0, train, target)
    # run(1, train, target)
    # run(2, train, target)
    run(3, train, target)
    run(4, train, target)

