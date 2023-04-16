from model import NaiveConvolutionNetwork
from utils_xvt import load_dataset,train_model
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

import modeling_xvt
import modeling_lightweight_xvt
import modeling_lightweight_xvt_v2
import configuration_xvt
import importlib
importlib.reload(modeling_xvt)
importlib.reload(modeling_lightweight_xvt)
importlib.reload(modeling_lightweight_xvt_v2)
importlib.reload(configuration_xvt)

# from modeling_xvt import XvtForImageClassification
# from modeling_lightweight_xvt import XvtForImageClassification
from modeling_lightweight_xvt_v2 import XvtForImageClassification  # added dropout layers
from configuration_xvt import XvtConfig
from configuration_xvt import XvtScheduler





def main():

    train_dataloader, val_dataloader, test_dataloader = load_dataset(batch_size=128)
    config = XvtConfig()
    args = XvtScheduler()
    model = XvtForImageClassification(config)

    train_model(model, train_dataloader, val_dataloader, test_dataloader, args, device='mps', use_weight_loss=True)

if __name__ == "__main__":
    main()
