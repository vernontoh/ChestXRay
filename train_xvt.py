from models.naive_cnn import NaiveConvolutionNetwork
from utils_xvt import load_dataset,train_model

# from XvT.modeling_lightweight_xvt import XvtForImageClassification
from XvT.modeling_lightweight_xvt_v2 import XvtForImageClassification  # added dropout layers
from XvT.configuration_xvt import XvtConfig
from XvT.configuration_xvt import XvtScheduler



def main():

    # indicating device to train with 
    device = "mps"

    # loading our training, validation and test set
    train_dataloader, val_dataloader, test_dataloader = load_dataset(batch_size=128)

    # initializing configs for model and scheduler
    config = XvtConfig()
    args = XvtScheduler()
    
    # intialize model with config
    model = XvtForImageClassification(config)

    # start training model
    train_model(model, train_dataloader, val_dataloader, test_dataloader, args, device=device, use_weight_loss=True)

if __name__ == "__main__":
    main()
