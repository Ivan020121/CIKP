import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import torchaudio.transforms as T

from imunet.resnet import resnet18
from imunet.mamba import HARMamba


def MinMaxScaler(data, return_scalers=False):
    """Min Max normalizer.

    Args:
      - data: original data

    Returns:
      - norm_data: normalized data
    """
    min = np.min(data, 0)
    max = np.max(data, 0)
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    norm_data = numerator / (denominator + 1e-7)
    if return_scalers:
        return norm_data, min, max
    return norm_data


def MinMaxArgs(data, min, max):
    """
    Args:
        data: given data
        min: given min value
        max: given max value

    Returns:
        min-max scaled data by given min and max
    """
    numerator = data - min
    denominator = max - min
    norm_data = numerator / (denominator + 1e-7)
    return norm_data


class STFTEmbedder(nn.Module):
    """
    STFT transformation
    """

    def __init__(self, device, seq_len, n_fft, hop_length):
        super().__init__()
        self.device = device
        self.seq_len = seq_len
        self.n_fft = n_fft
        self.hop_length = hop_length

    def cache_min_max_params(self, train_data):
        """
        Args:
            train_data: training timeseries dataset. shape: B*L*K
        this function initializes the min and max values for the real and imaginary parts.
        we'll use this function only once, before the training loop starts.
        """
        real, imag = self.stft_transform(train_data)
        # compute and cache min and max values
        real, min_real, max_real = MinMaxScaler(real.cpu().numpy(), True)
        imag, min_imag, max_imag = MinMaxScaler(imag.cpu().numpy(), True)
        self.min_real, self.max_real = torch.Tensor(min_real), torch.Tensor(max_real)
        self.min_imag, self.max_imag = torch.Tensor(min_imag), torch.Tensor(max_imag)


    def stft_transform(self, data):
        """
        Args:
            data: time series data. Shape: B*L*K
        Returns:
            real and imaginary parts of the STFT transformation
        """
        data = torch.permute(data, (0, 2, 1))  # we permute to match requirements of torchaudio.transforms.Spectrogram
        spec = T.Spectrogram(n_fft=self.n_fft, hop_length=self.hop_length, center=False, power=None).to(data.device)
        transformed_data = spec(data)
        return transformed_data.real, transformed_data.imag

    def ts2img(self, signal):
        assert self.min_real is not None, "use init_norm_args() to compute scaling arguments"
        # convert to complex spectrogram
        real, imag = self.stft_transform(signal)
        # MinMax scaling
        real = (MinMaxArgs(real, self.min_real.to(self.device), self.max_real.to(self.device)) - 0.5) * 2
        imag = (MinMaxArgs(imag, self.min_imag.to(self.device), self.max_imag.to(self.device)) - 0.5) * 2
        # stack real and imag parts
        stft_out = torch.cat((real, imag), dim=1)
        return stft_out


class STFTModel(nn.Module):
    """
    端到端 STFT 分类模型
    """

    def __init__(self,
                 seq_len: int,
                 n_fft: int,
                 hop_length: int,
                 num_classes: int,
                 device: str,
                 in_channels: int,):
        super(STFTModel, self).__init__()
        self.seq_len = seq_len
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.device = device
        self.in_channels = in_channels

        # STFT 转换层
        self.stft_embedder = STFTEmbedder(
            device=device,
            seq_len=seq_len,
            n_fft=n_fft,
            hop_length=hop_length
        )

        self.conv = resnet18(num_classes, include_top=False)

    def init_stft_embedder(self, train_loader):
        """使用训练数据初始化STFT归一化参数"""
        all_data = []
        for x, _, _ in train_loader:
            all_data.append(x)
        all_data = torch.cat(all_data, dim=0).to(self.device)
        self.stft_embedder.cache_min_max_params(all_data)

    def forward(self, x):
        x = self.stft_embedder.ts2img(x)
        x = self.conv(x)
        return x


class TFusion(nn.Module):
    def __init__(self, seq_len, n_fft, hop_length, device, in_channels, patch_size, stride, depth, num_classes):
        super(TFusion, self).__init__()
        self.stft_encoder = STFTModel(seq_len, n_fft, hop_length, num_classes, device, in_channels)
        self.signal_encoder = HARMamba(seq_size=seq_len, patch_size=patch_size, stride=stride, depth=depth, num_classes=num_classes, if_cls_token=False, c_in=in_channels)

        self.proj_head = nn.Linear(512 + 128, 128)
        self.proj_head_copy = nn.Linear(512 + 128, 128)
        self.proj_decoder = nn.Linear(128, 512 + 128)
        self.proj_classifier = nn.Linear(128, num_classes)

        self.classifier1 = nn.Linear(512 + 128, num_classes)
        self.classifier2 = nn.Linear(512 + 128 + 128, num_classes)

        # self.proj_head = nn.Linear(512 + 128, 256)
        # self.proj_head_copy = nn.Linear(512 + 128, 256)
        # self.proj_decoder = nn.Linear(256, 512 + 128)
        # self.proj_classifier = nn.Linear(256, num_classes)
        #
        # self.classifier1 = nn.Linear(512 + 128, num_classes)
        # self.classifier2 = nn.Linear(512 + 128 + 256, num_classes)

        self.relu = nn.ReLU()

    def freeze_encoder(self):
        """冻结两个 encoder（STFT 和 Signal）"""
        for m in [self.stft_encoder, self.signal_encoder]:
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

    def unfreeze_encoder(self):
        """解冻两个 encoder"""
        for m in [self.stft_encoder, self.signal_encoder]:
            m.train()
            for p in m.parameters():
                p.requires_grad = True

    def freeze_classifier1(self):
        """冻结两个分类头"""
        for p in self.classifier1.parameters():
            p.requires_grad = False
        self.classifier1.eval()

    def unfreeze_classifier1(self):
        """解冻两个分类头"""
        for p in self.classifier1.parameters():
            p.requires_grad = True
        self.classifier1.train()

    def freeze_classifier2(self):
        """冻结两个分类头"""
        for p in self.classifier2.parameters():
            p.requires_grad = False
        self.classifier2.eval()

    def unfreeze_classifier2(self):
        """解冻两个分类头"""
        for p in self.classifier2.parameters():
            p.requires_grad = True
        self.classifier2.train()

    def freeze_proj_head(self):
        """冻结投影头"""
        self.proj_head.eval()
        for p in self.proj_head.parameters():
            p.requires_grad = False

    def unfreeze_proj_head(self):
        """解冻投影头"""
        self.proj_head.train()
        for p in self.proj_head.parameters():
            p.requires_grad = True

    def freeze_proj_head_copy(self):
        """冻结投影头"""
        self.proj_head_copy.eval()
        for p in self.proj_head_copy.parameters():
            p.requires_grad = False

    def unfreeze_proj_head_copy(self):
        """解冻投影头"""
        self.proj_head_copy.train()
        for p in self.proj_head_copy.parameters():
            p.requires_grad = True

    def freeze_proj_decoder(self):
        """冻结投影头编码器"""
        self.proj_decoder.eval()
        for p in self.proj_decoder.parameters():
            p.requires_grad = False

    def unfreeze_proj_decoder(self):
        """解冻投影头编码器"""
        self.proj_decoder.train()
        for p in self.proj_decoder.parameters():
            p.requires_grad = True

    def freeze_proj_classifier(self):
        """冻结投影头编码器"""
        self.proj_classifier.eval()
        for p in self.proj_classifier.parameters():
            p.requires_grad = False

    def unfreeze_proj_classifier(self):
        """解冻投影头编码器"""
        self.proj_classifier.train()
        for p in self.proj_classifier.parameters():
            p.requires_grad = True

    def forward_copy(self, x):
        x1 = self.stft_encoder(x)
        x2 = self.signal_encoder(x, return_features=True)
        output = torch.cat([x1, x2], dim=1)
        return self.proj_head_copy(output).detach()

    def forward(self, x, mode):
        x1 = self.stft_encoder(x)
        x2 = self.signal_encoder(x, return_features=True)
        output = torch.cat([x1, x2], dim=1)

        # if mode == "stg1":
        #     return self.classifier1(output)
        # elif mode == "stg2":
        #     return F.mse_loss(output, self.proj_decoder(self.proj_head(output)))
        # elif mode == "stg3":
        #     output = self.proj_head(output)
        #     return output, self.proj_classifier(output)
        # elif mode == "stg4":
        #     x3 = self.proj_head(output)
        #     return self.classifier2(torch.cat([output, x3], dim=1)), self.classifier1(output), self.proj_classifier(x3)
        # elif mode == "emb":
        #     x3 = self.proj_head(output)
        #     return torch.cat([output, x3], dim=1)

        if mode == "stg1":
            return self.classifier1(output)
        elif mode == "stg2":
            return self.proj_head(output)
        elif mode == "stg3":
            x3 = self.proj_head(output)
            output = torch.cat([output, x3], dim=1)
            return self.classifier2(output)
        else:
            return None
