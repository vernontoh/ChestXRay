# ChestXRay


## Install required packages
```
pip install -r requirements.txt
```

## Downloading the dataset
The dataset used for training, validation and testing can be found here:  
https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345


## Model architectures
1. CNN
2. DenseNet
3. ResNet
4. U-Net
5. XvT (vision transformer)


## Loading our trained models
1. Download our trained models from https://drive.google.com/drive/folders/1EF_D6Dd4H3FiBGfJs65KMpSvEbrYqC-3
2. Place them within this project directory 
3. Run the line: ```torch.load('PATH/model_name.pt)```

Example: 
```
model = torch.load('Trained Models/ResNet.pt')
test_input = torch.randn(1, 3, 224, 224)
out = model(test_input)
print(out)
```

## Training XvT from scratch

```
python train_xvt.py --device mps \
        --num_layers 6 \
        --batch_size 128
```

Model and training configurations can be further adjusted by making changes in the `XvT/configuration_xvt.py` file.


## Training other models from scratch

1. In the main() function of train.py, define your model of choice
```
def main():
    train_dataloader, val_dataloader, test_dataloader = load_dataset(batch_size=64)
    model = ResNet()
    train_model(model, train_dataloader, val_dataloader, device='cuda', n_epochs=40, use_weight_loss=True)
```
2. Run train.py
```
python train.py
```