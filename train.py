from configuration_xvt import XvtScheduler
from models.naive_cnn import NaiveConvolutionNetwork
from models.resnet import ResNet
from models.densenet import DenseNet
from utils_xvt import load_dataset, train_model
import os
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from read_data import ChestXrayDataSet
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score,accuracy_score
from tqdm import tqdm


def main():
    train_dataloader, val_dataloader, test_dataloader = load_dataset(batch_size=64)
    args = XvtScheduler()
    model = DenseNet(
        growth_rate=16,
        block_config=(4, 8, 12, 8),
        n_init_features=32,
        bn_size=4,
        drop_rate=0,
        n_classes=14
    )
    # model = ResNet()
    # model = NaiveConvolutionNetwork()
    train_model(model, train_dataloader, val_dataloader, test_dataloader, args, device='cuda', use_weight_loss=True)

if __name__ == "__main__":
    main()
