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

class Informer(nn.Module):
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
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
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
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)

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
