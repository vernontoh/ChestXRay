from model import NaiveConvolutionNetwork
from utils import load_dataset,train_model
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
    
    train_dataloader,test_dataloader = load_dataset(batch_size=64)
    model = NaiveConvolutionNetwork()
    train_model(model,train_dataloader,test_dataloader,device='cuda',n_epochs=20)

if __name__ == "__main__":
    main()
