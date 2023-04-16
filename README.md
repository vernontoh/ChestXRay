# ChestXRay


## Install required packages
```
pip install -r requirements.txt
```

## Model architectures
1. CNN
2. DenseNet
3. ResNet
4. U-Net
5. XvT (vision transformer)


## Training XvT from scratch

```
python train_xvt.py --device mps \
        --num_layers 6 \
        --batch_size 128
```

Model and training configurations can be further adjusted by making changes in the `XvT/configuration_xvt.py` file.