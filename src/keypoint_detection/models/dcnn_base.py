
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys

sys.path.append("/home/nxhoang/HPE/src/keypoint_detection")

from modules.masking import TriangularCausalMask, ProbMask
from modules.informer_encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from modules.informer_decoder import Decoder, DecoderLayer
from modules.attn import FullAttention, ProbAttention, AttentionLayer
from modules.position_encoder import DataEmbedding
from modules.mlp_mixer import MLPMixer


class fDCNN(nn.Module):
    def __init__(self, enc_in, c_out,
                 factor=5, d_model=512, n_heads=8, e_layers=5, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda')):
        super(Informer, self).__init__()
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                        d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )



        self.mlp_x = MLPMixer(
            image_size=(16, 32),
            channels=24,
            patch_size=4,
            dim=512,
            depth=1,
            num_classes=17
        )

        self.mlp_y = MLPMixer(
            image_size=(16, 32),
            channels=24,
            patch_size=4,
            dim=512,
            depth=1,
            num_classes=17
        )


    def forward(self, x_enc,
                enc_self_mask=None):
        data_channel1 = x_enc[:, 0, :, :]
        data_channel1 = torch.transpose(data_channel1, 1, 2)

        data_channel2 = x_enc[:, 1, :, :]
        data_channel2 = torch.transpose(data_channel2, 1, 2)

        data_channel3 = x_enc[:, 2, :, :]
        data_channel3 = torch.transpose(data_channel3, 1, 2)

        enc_out1 = self.enc_embedding(data_channel1)
        enc_out2 = self.enc_embedding(data_channel2)
        enc_out3 = self.enc_embedding(data_channel3)

        out1, _ = self.encoder(enc_out1, attn_mask=enc_self_mask)
        out2, _ = self.encoder(enc_out2, attn_mask=enc_self_mask)
        out3, _ = self.encoder(enc_out3, attn_mask=enc_self_mask)

        out = torch.cat([out1.unsqueeze(0), out2.unsqueeze(0), out3.unsqueeze(0)])
        out = torch.permute(out, (1, 0, 2, 3))
        out = out.reshape(out.shape[0], -1, 16, 32)

        x_pred = self.mlp_x(out)
        y_pred = self.mlp_y(out)

        pred = torch.cat([x_pred.unsqueeze(0), y_pred.unsqueeze(0)], dim=0)
        pred = torch.permute(pred, (1, 2, 0))

        return pred




class GLU(nn.Module):
    def __init__(self, in_dim):
        super(GLU, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(in_dim, in_dim)

    def forward(self, x): #x size = [batch, chan, freq, frame]
        lin = self.linear(x.permute(0, 2, 3, 1)) #x size = [batch, freq, frame, chan]
        lin = lin.permute(0, 3, 1, 2) #x size = [batch, chan, freq, frame]
        sig = self.sigmoid(x)
        res = lin * sig
        return res


class ContextGating(nn.Module):
    def __init__(self, in_dim):
        super(ContextGating, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(in_dim, in_dim)

    def forward(self, x): #x size = [batch, chan, freq, frame]
        lin = self.linear(x.permute(0, 2, 3, 1)) #x size = [batch, freq, frame, chan]
        lin = lin.permute(0, 3, 1, 2) #x size = [batch, chan, freq, frame]
        sig = self.sigmoid(lin)
        res = x * sig
        return res


class Dynamic_conv2d (nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, bias=False, n_basis_kernels=4,
                 temperature=31, pool_dim='freq'):
        super(Dynamic_conv2d, self).__init__()

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.pool_dim = pool_dim

        self.n_basis_kernels = n_basis_kernels
        self.attention = attention2d(in_planes, self.kernel_size, self.stride, self.padding, n_basis_kernels,
                                     temperature, pool_dim)

        self.weight = nn.Parameter(torch.randn(n_basis_kernels, out_planes, in_planes, self.kernel_size, self.kernel_size),
                                   requires_grad=True)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(n_basis_kernels, out_planes))
        else:
            self.bias = None

        for i in range(self.n_basis_kernels):
            nn.init.kaiming_normal_(self.weight[i])

    def forward(self, x): #x size : [bs, in_chan, frames, freqs]
        if self.pool_dim in ['freq', 'chan']:
            softmax_attention = self.attention(x).unsqueeze(2).unsqueeze(4)    # size : [bs, n_ker, 1, frames, 1]
        elif self.pool_dim == 'time':
            softmax_attention = self.attention(x).unsqueeze(2).unsqueeze(3)    # size : [bs, n_ker, 1, 1, freqs]
        elif self.pool_dim == 'both':
            softmax_attention = self.attention(x).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)    # size : [bs, n_ker, 1, 1, 1]

        batch_size = x.size(0)

        aggregate_weight = self.weight.view(-1, self.in_planes, self.kernel_size, self.kernel_size) # size : [n_ker * out_chan, in_chan]

        if self.bias is not None:
            aggregate_bias = self.bias.view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding)
            # output size : [bs, n_ker * out_chan, frames, freqs]

        output = output.view(batch_size, self.n_basis_kernels, self.out_planes, output.size(-2), output.size(-1))
        # output size : [bs, n_ker, out_chan, frames, freqs]

        if self.pool_dim in ['freq', 'chan']:
            assert softmax_attention.shape[-2] == output.shape[-2]
        elif self.pool_dim == 'time':
            assert softmax_attention.shape[-1] == output.shape[-1]

        output = torch.sum(output * softmax_attention, dim=1)  # output size : [bs, out_chan, frames, freqs]

        return output


class attention2d(nn.Module):
    def __init__(self, in_planes, kernel_size, stride, padding, n_basis_kernels, temperature, pool_dim):
        super(attention2d, self).__init__()
        self.pool_dim = pool_dim
        self.temperature = temperature

        hidden_planes = int(in_planes / 4)

        if hidden_planes < 4:
            hidden_planes = 4

        if not pool_dim == 'both':
            self.conv1d1 = nn.Conv1d(in_planes, hidden_planes, kernel_size, stride=stride, padding=padding, bias=False)
            self.bn = nn.BatchNorm1d(hidden_planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv1d2 = nn.Conv1d(hidden_planes, n_basis_kernels, 1, bias=True)
            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        else:
            self.fc1 = nn.Linear(in_planes, hidden_planes)
            self.relu = nn.ReLU(inplace=True)
            self.fc2 = nn.Linear(hidden_planes, n_basis_kernels)

    def forward(self, x): #x size : [bs, chan, frames, freqs]
        if self.pool_dim == 'freq':
            x = torch.mean(x, dim=3)  #x size : [bs, chan, frames]
        elif self.pool_dim == 'time':
            x = torch.mean(x, dim=2)  #x size : [bs, chan, freqs]
        elif self.pool_dim == 'both':
            # x = torch.mean(torch.mean(x, dim=2), dim=1)  #x size : [bs, chan]
            x = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)
        elif self.pool_dim == 'chan':
            x = torch.mean(x, dim=1)  #x size : [bs, freqs, frames]

        if not self.pool_dim == 'both':
            x = self.conv1d1(x)               #x size : [bs, hid_chan, frames]
            x = self.bn(x)
            x = self.relu(x)
            x = self.conv1d2(x)               #x size : [bs, n_ker, frames]
        else:
            x = self.fc1(x)               #x size : [bs, hid_chan]
            x = self.relu(x)
            x = self.fc2(x)               #x size : [bs, n_ker]

        return F.softmax(x / self.temperature, 1)


class CNN(nn.Module):
    def __init__(self,
                 n_input_ch,
                 activation="Relu",
                 conv_dropout=0,
                 kernel=[3, 3, 3],
                 pad=[1, 1, 1],
                 stride=[1, 1, 1],
                 n_filt=[64, 64, 64],
                 pooling=[(1, 4), (1, 4), (1, 4)],
                 normalization="batch",
                 n_basis_kernels=4,
                 DY_layers=[0, 1, 1, 1, 1, 1, 1],
                 temperature=31,
                 pool_dim='freq'):
        super(CNN, self).__init__()
        self.n_filt = n_filt
        self.n_filt_last = n_filt[-1]
        cnn = nn.Sequential()

        def conv(i, normalization="batch", dropout=None, activ='relu'):
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

            if activ.lower() == "leakyrelu":
                cnn.add_module("Relu{0}".format(i), nn.LeakyReLu(0.2))
            elif activ.lower() == "relu":
                cnn.add_module("Relu{0}".format(i), nn.ReLu())
            elif activ.lower() == "glu":
                cnn.add_module("glu{0}".format(i), GLU(out_dim))
            elif activ.lower() == "cg":
                cnn.add_module("cg{0}".format(i), ContextGating(out_dim))

            if dropout is not None:
                cnn.add_module("dropout{0}".format(i), nn.Dropout(dropout))

        for i in range(len(n_filt)):
            conv(i, normalization=normalization, dropout=conv_dropout, activ=activation)
            cnn.add_module("pooling{0}".format(i), nn.AvgPool2d(pooling[i]))
        self.cnn = cnn

    def forward(self, x):    #x size : [bs, chan, frames, freqs]
        x = self.cnn(x)
        return x


class BiGRU(nn.Module):
    def __init__(self, n_in, n_hidden, dropout=0, num_layers=1):
        super(BiGRU, self).__init__()
        self.rnn = nn.GRU(n_in, n_hidden, bidirectional=True, dropout=dropout, batch_first=True, num_layers=num_layers)

    def forward(self, x):
        #self.rnn.flatten_parameters()
        x, _ = self.rnn(x)
        return x


class CRNN(nn.Module):
    def __init__(self,
                 n_input_ch,
                 n_class=10,
                 activation="glu",
                 conv_dropout=0.5,
                 n_RNN_cell=128,
                 n_RNN_layer=2,
                 rec_dropout=0,
                 attention=True,
                 **convkwargs):
        super(CRNN, self).__init__()
        self.n_input_ch = n_input_ch
        self.attention = attention
        self.n_class = n_class

        self.cnn = CNN(n_input_ch=n_input_ch, activation=activation, conv_dropout=conv_dropout, **convkwargs)
        self.rnn = BiGRU(n_in=self.cnn.n_filt[-1], n_hidden=n_RNN_cell, dropout=rec_dropout, num_layers=n_RNN_layer)

        self.dropout = nn.Dropout(conv_dropout)
        self.sigmoid = nn.Sigmoid()
        self.dense = nn.Linear(n_RNN_cell * 2, n_class)

        if self.attention:
            self.dense_softmax = nn.Linear(n_RNN_cell * 2, n_class)
            if self.attention == "time":
                self.softmax = nn.Softmax(dim=1)          # softmax on time dimension
            elif self.attention == "class":
                self.softmax = nn.Softmax(dim=-1)         # softmax on class dimension

    def forward(self, x): #input size : [bs, freqs, frames]
        #cnn
        if self.n_input_ch > 1:
            x = x.transpose(2, 3)
        else:
            x = x.transpose(1, 2).unsqueeze(1) #x size : [bs, chan, frames, freqs]
        x = self.cnn(x)
        bs, ch, frame, freq = x.size()
        if freq != 1:
            print("warning! frequency axis is large: " + str(freq))
            x = x.permute(0, 2, 1, 3)
            x = x.contiguous.view(bs, frame, ch*freq)
        else:
            x = x.squeeze(-1)
            x = x.permute(0, 2, 1) # x size : [bs, frames, chan]

        #rnn
        x = self.rnn(x) #x size : [bs, frames, 2 * chan]
        x = self.dropout(x)

        #classifier
        strong = self.dense(x) #strong size : [bs, frames, n_class]
        strong = self.sigmoid(strong)
        if self.attention:
            sof = self.dense_softmax(x) #sof size : [bs, frames, n_class]
            sof = self.softmax(sof) #sof size : [bs, frames, n_class]
            sof = torch.clamp(sof, min=1e-7, max=1)
            weak = (strong * sof).sum(1) / sof.sum(1) # [bs, n_class]
        else:
            weak = strong.mean(1)

        return strong.transpose(1, 2), weak



import torch.nn as nn
import torch
import cv2
from torchvision.transforms import Resize
from minirocket_3variables import fit, transform
from ChannelTrans import ChannelTransformer
from resblock import ResidualBlock
from CNN_spatio import Conv2D, CustomConv2D_layer1, CustomConv2D, regression, BiLSTMModel, TimeTransformer, \
    CustomConv2D_layer2, BiLSTMWithAttention
from sklearn.base import BaseEstimator, ClassifierMixin
from DynamicConv import Dynamic_conv2d
from SKConv import SKConv
import time


class posenet(nn.Module):
    """Our RoadSeg takes rgb and another (depth or normal) as input,
    and outputs freespace predictions.
    """

    def __init__(self):
        super(posenet, self).__init__()

        k1 = 4  # number branches of DyConv1
        k2 = 4  # number branches of DyConv2
        num_lay = 64  # number hidden dim of DyConv1
        D = 64  # number hidden dim of BiLSTM23
        N = 1  # number hidden layers of BiLSTm
        R = 32  # Reduction Ratios2234
        T = 34  # Temperature1
        hidden_reg = 32  # number hidden dim of Regression45
        self.tf = ChannelTransformer(vis=False, img_size=[17, 2], channel_num=128, num_layers=1, num_heads=3)
        self.rb1 = ResidualBlock(in_channels=1, out_channels=1, stride=1)
        self.rb2 = ResidualBlock(in_channels=1, out_channels=1, stride=1)
        self.CNN = Conv2D(in_channels=1, out_channels=1)
        self.CNN1 = CustomConv2D(in_channels=1, out_channels=32)
        self.CNN2 = CustomConv2D_layer1(in_channels=32 * 4, out_channels=4)
        self.CNN3 = CustomConv2D_layer2(in_channels=4, out_channels=4)
        self.D_conv = Dynamic_conv2d(in_planes=num_lay * 2, out_planes=num_lay * 2, kernel_size=7, ratio=(1 / R),
                                     stride=1, padding=3, dilation=1, groups=1, bias=True, K=4, temperature=T,
                                     init_weight=True)
        self.D_conv1 = Dynamic_conv2d(in_planes=1, out_planes=num_lay, kernel_size=7, ratio=(1 / R), stride=2,
                                      padding=1, dilation=1, groups=1, bias=True, K=k1, temperature=T, init_weight=True)
        self.D_conv2 = Dynamic_conv2d(in_planes=num_lay, out_planes=num_lay * 2, kernel_size=7, ratio=(1 / R), stride=2,
                                      padding=1, dilation=1, groups=1, bias=True, K=k2, temperature=T, init_weight=True)
        self.SK_conv1 = SKConv(features=1, M=4, G=1, r=34, stride=2, L=32)
        self.SK_conv2 = SKConv(features=1, M=4, G=1, r=34, stride=2, L=32)
        self.Att_BiLSTM = BiLSTMWithAttention(input_dim=1140, hidden_dim=64, num_layers=2, output_dim=17 * 2)
        self.regression = regression(input_dim=34 * (num_lay * 2), output_dim=34, hidden_dim=hidden_reg)
        # self.regression = regression(input_dim= 34*(10), output_dim = 34, hidden_dim= hidden_reg)
        self.BiLSTM = BiLSTMModel(input_dim=342, hidden_dim=D, num_layers=N,
                                  output_dim=17 * 2)  # good para 2-64 with 2CNN, 2-16 more better  with 3CNN
        self.ttf = TimeTransformer(input_dim=1140, hidden_dim=64, num_heads=3, num_layers=3, output_dim=17 * 2)

        self.decode = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True)
        )

        self.decode1 = nn.Sequential(
            nn.Conv2d(514, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True)
        )


        self.bn3 = nn.BatchNorm1d(2)

        self.bn = nn.BatchNorm2d(num_lay)
        self.bn1 = nn.BatchNorm2d(num_lay * 2)
        self.bn2 = nn.BatchNorm2d(num_lay * 2 + 10)

        self.relu = nn.ReLU(inplace=True)



    def forward(self, x_frame, x_sequence, batch):  # 16,2,3,114,32

        x1 = x_frame[:, 0:1, :, :]  # 1,114,32
        x2 = x_frame[:, 1:2, :, :]
        x3 = x_frame[:, 2:3, :, :]





        x1 = torch.transpose(x1, 1, 2)  # 16,2,114,3,32
        x1 = torch.flatten(x1, 2, 3)  # 16,2,114,96
        torch_resize = Resize([148, 14])
        x1 = torch_resize(x1)  # 136,32
        x1 = x1.view(batch, 1, 148, 14)

        x2 = torch.transpose(x2, 1, 2)  # 16,2,114,3,32
        x2 = torch.flatten(x2, 2, 3)  # 16,2,114,96
        torch_resize = Resize([148, 14])
        x2 = torch_resize(x2)
        x2 = x2.view(batch, 1, 148, 14)

        x3 = torch.transpose(x3, 1, 2)  # 16,2,114,3,32
        x3 = torch.flatten(x3, 2, 3)  # 16,2,114,96
        torch_resize = Resize([148, 14])
        x3 = torch_resize(x3)
        x3 = x3.view(batch, 1, 148, 14)



        # x1 = self.D_conv(x1) # dynamic
        x1 = self.D_conv1(x1)
        # x1 = self.SK_conv1(x1) # selective
        x1 = self.bn(x1)
        x1 = self.relu(x1)

        # x2 = self.D_conv(x2)
        # x2 = self.SK_conv1(x2)
        x2 = self.D_conv1(x2)
        x2 = self.bn(x2)
        x2 = self.relu(x2)

        # x3 = self.D_conv(x3)
        # x3 = self.SK_conv1(x3)
        x3 = self.D_conv1(x3)
        x3 = self.bn(x3)
        x3 = self.relu(x3)

        out1 = torch.cat([x1, x2, x3], dim=3)
        # out1 = self.D_conv2(out1)
        out1 = self.D_conv2(out1)
        # out1 = self.SK_conv2(out1)
        out1 = self.relu(out1)
        out1 = self.bn1(out1)

        # out1 = self.D_conv(out1)
        # out1 = self.relu(out1)
        # out1 = self.bn1(out1)
        " Max-Pooling"
        m = torch.nn.MaxPool2d((2, 3))
        out1 = m(out1)

        x = self.regression(out1)
        x = x.reshape(batch, 17, 2)
        return x


def weights_init(m):
    # classname = m.__class__.__name__
    # if classname.find('Conv') != -1:
    #     nn.init.xavier_normal_(m.weight.data)
    #     nn.init.xavier_normal_(m.bias.data)
    # elif classname.find('BatchNorm2d') != -1:
    #     nn.init.normal_(m.weight.data, 1.0)
    #     nn.init.constant_(m.bias.data, 0.0)
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
        # nn.init.xavier_normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

