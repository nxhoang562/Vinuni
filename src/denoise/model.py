from torch import nn
from torchsummary import summary


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # layer1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, padding=0)
        # layer2
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, padding=0)
        # layer3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, padding=0)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        # layer4
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.up_pool1 = nn.Upsample(scale_factor=2)
        # self.up_pool1 = nn.Upsample(scale_factor=2, mode ='nearest')

        # layer5
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.up_pool2 = nn.Upsample(scale_factor=2)

        # layer6
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.up_pool3 = nn.Upsample(scale_factor=2)

        # layer7
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()

    def forward(self, x):
        x = self.up_pool1(self.relu1(self.conv1(x)))
        x = self.up_pool2(self.relu2(self.conv2(x)))
        x = self.up_pool3(self.relu3(self.conv3(x)))
        x = self.relu4(self.conv4(x))

        return x


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == '__main__':
    test_model = AutoEncoder().to("cuda")

    # summary(test_model, input_size=(3, 32,136), batch_size=16, device="cuda")
    pass

# model = AutoEncoder().to("cuda")
#
# summary(model, (3, 32, 136))