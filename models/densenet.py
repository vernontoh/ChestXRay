from collections import OrderedDict
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(Transition, self).__init__()
        self.norm = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)


class DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, dropout_rate):
        super(DenseLayer, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input):
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        concat_features = torch.cat(prev_features, 1)
        bottleneck_output = self.conv1(self.relu(self.norm1(concat_features)))
        next_features = self.conv2(self.relu(self.norm2(bottleneck_output)))
        next_features = self.dropout(next_features)
        return next_features
    

class DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, dropout_rate=0):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layer = DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                dropout_rate=dropout_rate
            )
            layers.append(layer)

        self.dense_layers = nn.Sequential(*layers)

    def forward(self, init_features):
        features = [init_features]
        for layer in self.dense_layers:
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)
    

class DenseNet(nn.Module):
    def __init__(self, growth_rate=24, block_config=(4, 8, 16, 12),
                 n_init_features=48, bn_size=4, dropout_rate=0, n_classes=14):
        super(DenseNet, self).__init__()

        # First conv layer
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, n_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(n_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        
        n_features = n_init_features
        for i, num_layers in enumerate(block_config):   # Add Dense blocks based on input parameters
            block = DenseBlock(
                num_layers=num_layers,
                num_input_features=n_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                dropout_rate=dropout_rate
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            n_features = n_features + num_layers * growth_rate
            if i != len(block_config) - 1:   # Add transition layer after each denseblock except last
                trans = Transition(num_input_features=n_features,
                                    num_output_features=n_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                n_features = n_features // 2

        self.features.add_module('norm5', nn.BatchNorm2d(n_features))

        self.classifier = nn.Linear(n_features, n_classes)

        # Use the official initialization method from PyTorch
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = nn.ReLU()(features)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out