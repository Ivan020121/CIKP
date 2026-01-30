import torch
import torch.nn as nn


class MSC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MSC, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1)

        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # first scale
        f1 = self.relu(self.bn1(self.conv1(x)))  # (B*J, 32, T1)

        # second scale
        f2 = self.relu(self.bn2(self.conv2(f1)))  # (B*J, 32, T2)

        # third scale
        f3 = self.relu(self.bn3(self.conv3(f2)))  # (B*J, 32, T3)

        return f1, f2, f3


# class MSC(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(MSC, self).__init__()
#
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=2, padding=3)
#         self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=5, stride=2, padding=1)
#         self.conv3 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1)
#
#         self.conv2d_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), stride=(1, 1), padding=0)
#         self.conv2d_2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 1), stride=(1, 1), padding=0)
#         self.conv2d_3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 1), stride=(1, 1), padding=0)
#         self.pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
#
#         self.bn1 = nn.BatchNorm1d(out_channels)
#         self.bn2 = nn.BatchNorm1d(out_channels)
#         self.bn3 = nn.BatchNorm1d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):  # (B, 17, 2, 50)
#         B, N, C, T = x.size()
#
#         # first scale
#         f1_1d = self.conv1(x.reshape(B*N, C, T))
#         f1_2d = self.conv2d_1(x.permute(0, 2, 1, 3))
#         f1_2d = self.pool(f1_2d)
#         f1_2d = f1_2d.permute(0, 2, 1, 3)
#         f1_2d = f1_2d.reshape(B*N, self.out_channels, -1)
#         f1 = self.relu(self.bn1(f1_1d + f1_2d))  # (B*J, 32, T1)
#
#         # second scale
#         f2_1d = self.conv2(f1)
#         f2_2d = self.conv2d_2(f1.reshape(B, N, self.out_channels, -1).permute(0, 2, 1, 3))
#         f2_2d = self.pool(f2_2d)
#         f2_2d = f2_2d.permute(0, 2, 1, 3)
#         f2_2d = f2_2d.reshape(B * N, self.out_channels, -1)
#         f2 = self.relu(self.bn2(f2_1d + f2_2d))  # (B*J, 32, T1)
#
#         # third scale
#         f3_1d = self.conv3(f2)
#         f3_2d = self.conv2d_3(f2.reshape(B, N, self.out_channels, -1).permute(0, 2, 1, 3))
#         f3_2d = self.pool(f3_2d)
#         f3_2d = f3_2d.permute(0, 2, 1, 3)
#         f3_2d = f3_2d.reshape(B * N, self.out_channels, -1)
#         f3 = self.relu(self.bn3(f3_1d + f3_2d))  # (B*J, 32, T1)
#
#         return f1, f2, f3
