import numpy as np
import torch
import torch.nn as nn
import torchvision

class ETFClassifier(nn.Module):
    def __init__(self, n_features, n_classes):
        super(ETFClassifier, self).__init__()

        # Ensure n_features >= n_classes
        if n_features < n_classes:
            raise ValueError("Number of features must be greater than or equal to the number of classes")

        # Create a random rotation matrix U
        U, _, _ = torch.svd(torch.randn(n_features, n_classes))

        # Create the ETF matrix E
        K = n_classes
        identity_K = torch.eye(K)
        ones_K = torch.ones(K, K)
        E = np.sqrt(K / (K - 1)) * U @ (identity_K - (1/K) * ones_K)

        # Store as a parameter of the classifier
        self.classifier = torch.nn.Parameter(E, requires_grad=False)


    def forward(self, x):
        # Apply the classifier to the input
        return torch.mm(x, self.classifier)

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

        self.temperature = 0.0
        self.mode = False

    def set_temperature(self, temperature):
        self.temperature = temperature

    def set_mode(self, mode):
        self.mode = mode

    def forward(self, x):
        batch_size, C, height, width = x.size()
        query_out = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key_out = self.key_conv(x).view(batch_size, -1, width * height)

        attention = self.softmax(torch.bmm(query_out, key_out))
        value_out = self.value_conv(x).view(batch_size, -1, width * height)

        out = torch.bmm(value_out, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, height, width)

        if self.mode:
            out = x + out * self.gamma
        else:
            out = x + out * self.temperature

        return out


class ResNet_FE(nn.Module):
    def __init__(self):
        super().__init__()
        model_resnet = torchvision.models.resnet50(True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4

        num_channels = self.layer4[-1].conv3.out_channels
        self.self_attn = SelfAttention(num_channels)

        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu,
                                            self.maxpool, self.layer1,
                                            self.layer2, self.layer3,
                                            self.layer4, self.self_attn, self.avgpool)
        self.bottle = nn.Linear(2048, 256)
        self.bn = nn.BatchNorm1d(256)

    def forward(self, x):
        out = self.feature_layers(x)
        out = out.view(out.size(0), -1)
        out = self.bn(self.bottle(out))
        return out

