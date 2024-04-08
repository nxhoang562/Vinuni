import torch
from torch import nn
import sys
from torchsummary import summary
from src.denoise.model import AutoEncoder
from src.keypoint_detection.models.fdcnn import CNN

sys.path.append("/home/nxhoang/worl/HPE")

from src.keypoint_detection.modules.FDCNN import Dynamic_conv2d

class CombinedModel(nn.Module):
    def __init__(self, denoiser, predictor):
        super(CombinedModel, self).__init__()
        # Denoiser
        self.denoiser = denoiser
        # FD CNN
        self.predictor = predictor

    def forward(self, x):
        with torch.no_grad():
            encoded = self.denoiser(x)
        output = self.predictor(encoded)
        return output



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AE = AutoEncoder()
CNN = CNN()
model = CombinedModel(AE.encoder, CNN)
model = model.to(device)
summary(model, (3, 32, 136))






