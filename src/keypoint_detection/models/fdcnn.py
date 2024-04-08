import torch
import torch.nn as nn
from torchsummary import summary

import sys

sys.path.append("/home/nxhoang/HPE/src/keypoint_detection")

from modules.FDCNN import Dynamic_conv2d


class CNN(nn.Module):

    def __init__(self, n_input_ch= 3,  # no. channels
                 activation='Relu',
                 conv_dropout=0,
                 kernel=[3, 3, 3],  # kernel size
                 pad=[1, 1, 1],
                 stride=[1, 1, 1],
                 # n_filt=[64, 64, 64],  # no. filters for each conv layer
                 n_filt=[8, 8, 8],
                 # pooling=[(1, 4), (1, 4), (1, 4)],  # pooling dimensions for each poooling layers
                 pooling=[(1, 2), (1, 2), (1, 2)],  # pooling dimensions for each poooling layers
                 normalization="batch",
                 n_basis_kernels=4,  # no. kernels
                 DY_layers=[1, 1, 1, 1, 1, 1],  # where a dynamic layer is used
                 temperature = 31,
                 pool_dim='freq'):  # dimension for pooling
        super(CNN, self).__init__()
        self.n_filt = n_filt
        self.n_filt_last = n_filt[-1]
        cnn = nn.Sequential()

        def conv(i, normalization='batch', dropout=None, activ='relu'):
            in_dim = n_input_ch if i == 0 else n_filt[i - 1]
            out_dim = n_filt[i]
            if DY_layers[i] == 1:
                cnn.add_module("conv{0}".format(i), Dynamic_conv2d(in_dim, out_dim, kernel[i], stride[i], pad[i],
                                                                   n_basis_kernels=n_basis_kernels,
                                                                   temperature=temperature, pool_dim=pool_dim))
            else:
                cnn.add_module("conv{0}".format(i), nn.Conv2d(in_dim, out_dim, kernel[i], stride[i], pad[i]))

            if normalization == "batch":
                cnn.add_module("batchnorm{0}".format(i), nn.BatchNorm2d(out_dim, eps=0.001, momentum=0.99))

            elif normalization == "layer":
                cnn.add_module("layernorm{0}".format(i), nn.GroupNorm(1, out_dim))


            elif activ.lower() == "relu":
                cnn.add_module("Relu{0}".format(i), nn.ReLu())

            if dropout is not None:
                cnn.add_module("dropout{0}".format(i), nn.Dropout(dropout))

        for i in range(len(n_filt)):
            conv(i, normalization=normalization, dropout=conv_dropout, activ=activation)
            cnn.add_module("pooling{0}".format(i), nn.AvgPool2d(pooling[i]))
        self.cnn = cnn

        self.fc1 = nn.Linear(in_features=64, out_features=128)
        # self.fc1 = nn.Linear(in_features=4352, out_features=128)
        self.batch_norm = nn.BatchNorm1d(num_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=34)
        self.drop = nn.Dropout(p=0.1)

    def forward(self, x):  # x size : [batch, channel, frames, freqs] 32, 3, 10,114
        batch = x.shape[0]
        # print(x.shape)
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.fc1(x)
        x = self.batch_norm(x)
        x = nn.functional.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = x.reshape(batch, 17, 2)

        return x


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = CNN()
# model = model.to(device)
# summary(model, (3,4,17))

# shape input: [64,3,136,32] = [batch size, channels, freq, time]
