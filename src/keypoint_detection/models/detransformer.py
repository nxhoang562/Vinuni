from torch import nn

import sys
sys.path.append("/home/nxhoang/HPE")
from src.keypoint_detection.modules.position_encoder import generate_original_PE, generate_regular_PE
from src.keypoint_detection.modules.transformer_encoder import Encoder


class DeTransformer(nn.Module):
    def __init__(self, denoise_encoder,
                 d_input, d_model, d_output, q, v, h, N, attention_size, dropout=0.3, chunk_mode="chunk", pe=None,
                 pe_period=None):
        super().__init__()

        # Denoiser
        self.denoise_encoder = denoise_encoder
        for param in self.denoise_encoder.parameters():
            param.requires_grad = True

        # Transformer
        self.transformer_layers = nn.ModuleList([
            Encoder(d_model, q, v, h, attention_size=attention_size, dropout=dropout, chunk_mode=chunk_mode) for _ in range(N)
        ])
        self.pe_period = pe_period
        self.d_model = d_model

        self.embedding = nn.Linear(d_input, d_model)
        self.cnn = nn.Conv1d(32, 17, kernel_size=3, padding=1)
        self.linear = nn.Linear(d_model, d_output)
        self.batch_norm = nn.BatchNorm2d(17)
        self.relu = nn.ReLU()

        pe_functions = {
            'original': generate_original_PE,
            'regular': generate_regular_PE,
        }

        if pe in pe_functions.keys():
            self._generate_PE = pe_functions[pe]
            self._pe_period = pe_period
        elif pe is None:
            self._generate_PE = None
        else:
            raise NameError(
                f'PE "{pe}" not understood. Must be one of {", ".join(pe_functions.keys())} or None.')

    def forward(self, x):
        # x = self.denoise_encoder(x)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(x.shape[0], x.shape[1], -1)

        K = x.shape[1]
        x = self.embedding(x)

        if self._generate_PE is not None:
            positional_encoding = self._generate_PE(K, self.d_model)
            positional_encoding = positional_encoding.to(x.device)
            x.add_(positional_encoding)

        for layer in self.transformer_layers:
            x = layer(x)

        x = self.relu(self.cnn(x))
        output = self.linear(x)

        return output
