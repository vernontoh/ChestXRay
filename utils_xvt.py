import os
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from read_data import ChestXrayDataSet
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score,accuracy_score
from tqdm import tqdm
from timm.scheduler import create_scheduler




DATA_DIR = 'ChestX-ray14/images'
TEST_IMAGE_LIST = 'ChestX-ray14/labels/test_list.txt'
VAL_IMAGE_LIST = 'ChestX-ray14/labels/val_list.txt'
TRAIN_IMAGE_LIST = 'ChestX-ray14/labels/train_list.txt'
N_CLASSES = 14
CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']


        
def load_dataset(batch_size=64):
    print("LOADING DATASET")
    trainTransforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    valTransforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    testTransforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_dataset = ChestXrayDataSet(data_dir=DATA_DIR,
                                    image_list_file=TEST_IMAGE_LIST,
                                    transform=testTransforms)
    
    val_dataset = ChestXrayDataSet(data_dir=DATA_DIR,
                                    image_list_file=VAL_IMAGE_LIST,
                                    transform=valTransforms)
                                   
    train_dataset = ChestXrayDataSet(data_dir=DATA_DIR,
                                    image_list_file=TRAIN_IMAGE_LIST,
                                    transform=trainTransforms)
    
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,shuffle=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size,shuffle=False)
    train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
    
    return train_loader, val_loader, test_loader


def train_model(model, train_dataloader, val_dataloader, args, use_weight_loss=True):
    print("TRAINING START: ")
    pos_weight = torch.tensor([9.719982661465107,
                                 40.447330447330444,
                                 8.425640640264522,
                                 5.6423934376729905,
                                 19.512704490080054,
                                 17.73208919816543,
                                 82.86770140428678,
                                 21.162702906757268,
                                 24.023998285836726,
                                 48.68432479374729,
                                 44.56279809220986,
                                 66.5005931198102,
                                 33.122599704579024,
                                 493.92070484581495])
    if use_weight_loss:
        criterion =  torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = torch.nn.BCEWithLogitsLoss()

    # intialize our optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00025, weight_decay=0.05)
    lr_scheduler, _ = create_scheduler(args, optimizer=optimizer)
    lr_scheduler.step(0)

    running_loss = 0
    
    model = model.to(args.device)
    criterion = criterion.to(args.device)
    step = 0

    # writer will output to ./runs/ directory by default
    writer = SummaryWriter()
    
    # training loop
    for i in range(args.epochs):
        model.train()
        print(f"Epoch{i+1} / {args.epochs}:")
        for inputs, labels in tqdm(train_dataloader):
    
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            
            optimizer.zero_grad()
            
            logits = model(inputs)
            loss = criterion(logits,labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
           
            step += 1
                
            if (step + 1) % 100 == 0:
                print(f"Training loss at {step + 1} is :  {running_loss / 100}")
                writer.add_scalar('Loss/train', running_loss / 100, step + 1)
                running_loss = 0
        
        torch.save(model,f'model-checkpoint-{i + 1}.pt')

        eval_dict  = evaluate(model, val_dataloader, criterion, args.device)
        
        print(f"Eval loss at epoch {i+1} is:{eval_dict['loss']}")
        print(f"Eval accuracy at {i+1} is:{eval_dict['acc']}")
        print(f"Eval f1 score at {i+1} is:{eval_dict['f1']}")
        writer.add_scalar('Loss/valid', eval_dict['loss'], i + 1)
        
        ## Compute AUC
        AUROCs = compute_AUCs(eval_dict['labels'], eval_dict['logits'])
        AUROC_avg = np.array(AUROCs).mean()
        writer.add_scalar('AUROCs', AUROC_avg, i + 1)
        print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
        for n in range(N_CLASSES):
            print('The AUROC of {} is {}'.format(CLASS_NAMES[n], AUROCs[n]))     

        # Reduce learning rate if plateau
        lr_scheduler.step(epoch=i+1)              

        # Tensorboard logs
                        
                    
        
def compute_AUCs(gt, pred):
    """Computes Area Under the Curve (AUC) from prediction scores.

    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.

    Returns:
        List of AUROCs of all classes.
    """
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(N_CLASSES):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs



def evaluate(model, val_dataloader, criterion, device):
    
    print("----- Evaluating ------")
    
    model.eval()
    acc = torch.empty(14)
    all_predictions = torch.tensor([])
    all_labels = torch.tensor([])
    all_logits = torch.tensor([])
    running_loss = 0
    model = model.to(device)
    criterion = criterion.to(device)
    with torch.no_grad(): ## Disable gradient
        for batch in tqdm(val_dataloader):

            inputs,labels = batch
            
            inputs = inputs.to(device)
            labels = labels.to(device)
                
            logits = model(inputs)
           
            
            loss = criterion(logits,labels)

            probs = torch.sigmoid(logits)
            pred = (probs>0.5).float().cpu()
            
            labels = labels.cpu()
            all_logits = torch.cat((all_logits,logits.cpu()),axis=0)
            all_predictions = torch.cat((all_predictions,pred),axis=0)
            all_labels = torch.cat((all_labels,labels),axis=0)

            running_loss +=loss.item() 

   
    running_loss /= len(val_dataloader)
    acc = accuracy_score(all_predictions,all_labels)
    f1 = f1_score(all_predictions,all_labels,average=None)
    
    return {"pred":all_predictions,"acc":acc,"loss":running_loss,"labels":all_labels,"f1":f1,"logits":all_logits}
