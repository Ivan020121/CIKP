import torch
import torch.nn as nn
from torch.nn import functional as F

from kpnet.st_gcn import STGCN
from kpnet.msc import MSC
from kpnet.seblock import SEBlock


class MSTGCN(nn.Module):
    def __init__(self, in_channels, out_channels, num_class, graph_args, edge_importance_weighting, **kwargs):
        super(MSTGCN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_class = num_class

        self.graph_args = graph_args
        self.edge_importance_weighting = edge_importance_weighting

        self.msc = MSC(in_channels, out_channels)
        self.stgcn0 = STGCN(in_channels, num_class, graph_args, edge_importance_weighting)
        self.stgcn1 = STGCN(out_channels, num_class, graph_args, edge_importance_weighting)
        self.stgcn2 = STGCN(out_channels, num_class, graph_args, edge_importance_weighting)
        self.stgcn3 = STGCN(out_channels, num_class, graph_args, edge_importance_weighting)
        self.seblock = SEBlock(4 * 256)

        self.proj_head = nn.Linear(4 * 256, 128)
        self.proj_head_copy = nn.Linear(4 * 256, 128)
        self.proj_decoder = nn.Linear(128, 4 * 256)
        self.proj_classifier = nn.Linear(128, num_class)

        self.classifier1 = nn.Linear(4 * 256, num_class)
        self.classifier2 = nn.Linear(4 * 256 + 128, num_class)

        # self.proj_head = nn.Linear(4 * 256, 256)
        # self.proj_head_copy = nn.Linear(4 * 256, 256)
        # self.proj_decoder = nn.Linear(256, 4 * 256)
        # self.proj_classifier = nn.Linear(256, num_class)
        #
        # self.classifier1 = nn.Linear(4 * 256, num_class)
        # self.classifier2 = nn.Linear(4 * 256 + 256, num_class)

        self.relu = nn.ReLU(inplace=True)

    def freeze_encoder(self):
        for m in [self.msc, self.stgcn0, self.stgcn1, self.stgcn2, self.stgcn3, self.seblock]:
            for p in m.parameters():
                p.requires_grad = False

    def unfreeze_encoder(self):
        for m in [self.msc, self.stgcn0, self.stgcn1, self.stgcn2, self.stgcn3, self.seblock]:
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
        self.proj_head.eval()
        for p in self.proj_head.parameters():
            p.requires_grad = False

    def unfreeze_proj_head(self):
        self.proj_head.train()
        for p in self.proj_head.parameters():
            p.requires_grad = True

    def freeze_proj_head_copy(self):
        self.proj_head_copy.eval()
        for p in self.proj_head_copy.parameters():
            p.requires_grad = False

    def unfreeze_proj_head_copy(self):
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
        N, C, T, V, M = x.size()  # (N, 2, T, 17, 1)
        g0 = self.stgcn0(x)
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M * V, C, T)  # (N*M, 2, 17, T)
        f1, f2, f3 = self.msc(x)  # (N*M, C, 17, Tn)

        f1 = f1.view(N, M, V, self.out_channels, -1)
        f1 = f1.permute(0, 3, 4, 2, 1).contiguous()
        f2 = f2.view(N, M, V, self.out_channels, -1)
        f2 = f2.permute(0, 3, 4, 2, 1).contiguous()
        f3 = f3.view(N, M, V, self.out_channels, -1)
        f3 = f3.permute(0, 3, 4, 2, 1).contiguous()

        g1 = self.stgcn1(f1)
        g2 = self.stgcn2(f2)
        g3 = self.stgcn3(f3)

        out = torch.cat([g0, g1, g2, g3], dim=1)
        out = self.seblock(out)
        return self.proj_head_copy(out).detach()

    def forward(self, x, mode):
        N, C, T, V, M = x.size() # (N, 2, T, 17, 1)
        g0 = self.stgcn0(x)
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M * V, C, T) # (N*M, 2, 17, T)
        f1, f2, f3 = self.msc(x) # (N*M, C, 17, Tn)

        f1 = f1.view(N, M, V, self.out_channels, -1)
        f1 = f1.permute(0, 3, 4, 2, 1).contiguous()
        f2 = f2.view(N, M, V, self.out_channels, -1)
        f2 = f2.permute(0, 3, 4, 2, 1).contiguous()
        f3 = f3.view(N, M, V, self.out_channels, -1)
        f3 = f3.permute(0, 3, 4, 2, 1).contiguous()

        g1 = self.stgcn1(f1)
        g2 = self.stgcn2(f2)
        g3 = self.stgcn3(f3)

        out = torch.cat([g0, g1, g2, g3], dim=1)
        out = self.seblock(out)

        # if mode == "stg1":
        #     return self.classifier1(out)
        # elif mode == "stg2":
        #     return F.mse_loss(out, self.proj_decoder(self.proj_head(out)))
        # elif mode == "stg3":
        #     out = self.proj_head(out)
        #     return out, self.proj_classifier(out)
        # elif mode == "stg4":
        #     x3 = self.proj_head(out)
        #     return self.classifier2(torch.cat([out, x3], dim=1)), self.classifier1(out), self.proj_classifier(x3)
        # elif mode == "emb":
        #     x3 = self.proj_head(out)
        #     return torch.cat([out, x3], dim=1)

        if mode == "stg1":
            return self.classifier1(out)
        elif mode == "stg2":
            return self.proj_head(out)
        elif mode == "stg3":
            x3 = self.proj_head(out)
            out = torch.cat([out, x3], dim=1)
            return self.classifier2(out)
        else:
            return None
