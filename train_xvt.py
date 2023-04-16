from utils_xvt import load_dataset,train_model
import argparse

# from XvT.modeling_lightweight_xvt import XvtForImageClassification
from XvT.modeling_lightweight_xvt_v2 import XvtForImageClassification  # added dropout layers
from XvT.configuration_xvt import XvtConfig



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", default="cpu")
    parser.add_argument("-l", "--num_layers", type=int, default=6)
    parser.add_argument("-b", "--batch_size", type=int, default=128)
    args = parser.parse_args()

    # loading our training, validation and test set
    train_dataloader, val_dataloader, test_dataloader = load_dataset(args.batch_size)

    # initializing configs for model and scheduler
    config = XvtConfig(**vars(args))
    
    # intialize model with config
    model = XvtForImageClassification(config)

    # start training model
    train_model(model, train_dataloader, val_dataloader, config, use_weight_loss=True)

if __name__ == "__main__":
    main()
