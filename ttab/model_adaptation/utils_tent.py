import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import DBSCAN, HDBSCAN
import torchvision.models as models
from ttab.loads.models.resnet import (
    ResNetCifar,
    ResNetImagenet,
    ResNetMNIST,
    ViewFlatten,
)
from ttab.loads.models.wideresnet import WideResNet
import copy
from finch import FINCH
"""optimization dynamics"""

class ClusterNorm2d(nn.Module):
    def __init__(
        self, num_channels, eps=1e-5, momentum=0.1, threshold=1, affine=True
    ):
        super(ClusterNorm2d, self).__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.threshold = threshold
        # print(self.threshold )
        self.CNState = True  # 存储原始阈值
        self.affine = affine
        self._bn = nn.BatchNorm2d(
            num_channels, eps=eps, momentum=momentum, affine=affine
        )


    def _adaptive_clustering(self, x, mu_b, sigma2_b):
        b, c, h, w = x.size()
        # mu_b = mu_b.view(1, c, 1, 1)
        # sigma2_b = sigma2_b.view(1, c, 1, 1)    
        
        # print("[INFO] clusting")
        
        # 计算IN的均值和方差
        sigma2, mu = torch.var_mean(x, dim=[2, 3], keepdim=True, unbiased=True) 
        # print(f"[INFO] mu_b shape: {mu_b.shape}")
        # print(f"[INFO] mu shape: {mu.shape}")
        
        # 计算IN减去BN的结果
        # x_diff = (mu - mu_b) * torch.rsqrt(sigma2_b + self.eps)
        x_diff = mu
        # x_diff = (mu - mu_b) 
        # 转换成适合聚类的形式
        x_diff_flattened = x_diff.view(-1, self.num_channels)
        # print(f"[INFO] x_diff_flattened shape: {x_diff_flattened.shape}")

        # 聚类
        # print(f"[INFO] s_mu.mean(1): {s_mu.mean().item()}")
        data = x_diff_flattened.cpu().detach().numpy()

        # dataT = data.T
        # print(f"[INFO] x_diff_flattened: {x_diff_flattened.shape}")
        # print(f"[INFO] data: {data.shape}")
        
        c1, num_clust, req_c = FINCH(data, verbose=False)
        labels = c1
        
        # clustering = HDBSCAN().fit(x_diff_flattened.cpu().detach().numpy())
        # clustering = DBSCAN(s_mu.mean().item()*2).fit(x_diff_flattened.cpu().detach().numpy())
        # clustering = DBSCAN(((self.k*s_mu).mean(1).item())).fit(x_diff_flattened.cpu().detach().numpy())
        # 获取聚类结果
        # labels = clustering.labels_
        # 计算每个簇的统计值
        # unique_labels = np.unique(c1,axis=0)
        # print(f"[INFO] labels: {labels}")
        
        # print(f"[INFO] unique_labels: {unique_labels}")
        # print(f"[INFO] c1.shape: {c1.shape}")
        # print(f"[INFO] c1: {c1}")
        # print(f"[INFO] req_c: {req_c}")
        unique_labels, inverse_indices = np.unique(labels, axis=0, return_inverse=True)
        # flat_labels = np.arange(len(unique_labels))[inverse_indices]

        # 输出一维标签
        # print("Batch Clustering Results:")
        # print(flat_labels)
         
        # cluster_stats = []
        for label in unique_labels:
            
            cluster_indices = np.where(np.all(labels == label, axis=1))
            cluster_data = x[cluster_indices]  # 使用当前簇的数据            
            
            cluster_sigma2, cluster_mu = torch.var_mean(x[cluster_indices], dim=[0, 2, 3], keepdim=True, unbiased=True)
            # cluster_sigma2i, cluster_mui = torch.var_mean(cluster_data, dim=[2, 3], keepdim=True, unbiased=True) 
            # print(f"[INFO] cluster_indices: {cluster_indices}")
            # print(f"[INFO] cluster_indices: {labels == label}")

            sigma2_expanded = sigma2_b.repeat(cluster_data.shape[0],1,1,1) 
            mu_expanded = mu_b.repeat(cluster_data.shape[0],1,1,1) 
            # cluster_sigma2, cluster_mu = torch.var_mean(cluster_data, dim=[0, 2, 3], keepdim=True, unbiased=True)
            # x[cluster_indices] = (x[cluster_indices] - (mu_expanded)) * torch.rsqrt((sigma2_expanded) + self.eps)
            x[cluster_indices] = (x[cluster_indices] - (0.2*cluster_mu+0.8*mu_expanded)) * torch.rsqrt((0.2*cluster_sigma2+0.8*sigma2_expanded) + self.eps)
            # x[cluster_indices] = (x[cluster_indices] - (0.15*cluster_mu+0.85*mu_expanded)) * torch.rsqrt((0.15*cluster_sigma2+0.85*sigma2_expanded) + self.eps)
            # x[cluster_indices] = (x[cluster_indices] - (0.33*cluster_mu+0.66*mu_expanded)) * torch.rsqrt((0.33*cluster_sigma2+0.66*sigma2_expanded) + self.eps)
            
        return x

    def forward(self, x):
        b, c, h, w = x.size()
        # IN
        
        # sigma2, mu = torch.var_mean(x, dim=[2, 3], keepdim=True, unbiased=True) 

        if self.training:
            _ = self._bn(x)
            sigma2_b, mu_b = torch.var_mean(
                x, dim=[0, 2, 3], keepdim=True, unbiased=True
            )
        else:
            # if (
            #     self._bn.track_running_stats == False
            #     # and self._bn.running_mean is None
            #     # and self._bn.running_var is Nones
            # ):  # use batch stats
            sigma2_b, mu_b = torch.var_mean(
                x, dim=[0, 2, 3], keepdim=True, unbiased=True
            )
            # else:
                # # BN
                # mu_b = self._bn.running_mean.view(1, c, 1, 1)
                # sigma2_b = self._bn.running_var.view(1, c, 1, 1)
        if not self.CNState:
            mu_adj = mu_b
            sigma2_adj = sigma2_b
            x_n = (x - mu_adj) * torch.rsqrt(sigma2_adj + self.eps)
        else:
            # 使用自适应聚类获取归一化
        # print()
            mu_s = self._bn.running_mean.view(1, c, 1, 1)
            sigma2_s = self._bn.running_var.view(1, c, 1, 1)
            
            
            x_n = self._adaptive_clustering(x, mu_s, sigma2_s)
            del x
            # sigma2_adj, mu_adj = torch.var_mean(
                    # x_n, dim=[0, 2, 3], keepdim=True, unbiased=True
                # )
            # 使用聚类后的簇内统计值作为归一化参数
        #     mu_adj = torch.cat([stats[0] for stats in cluster_stats], dim=0)
        #     sigma2_adj = torch.cat([stats[1] for stats in cluster_stats], dim=0)

        # x_n = (x_n - mu_adj) * torch.rsqrt(sigma2_adj + self.eps)
        if self.affine:
            weight = self._bn.weight.view(1, c, 1, 1)
            bias = self._bn.bias.view(1, c, 1, 1)
            # x_n = (x_n*(1+sigma2_s/(h*w)))* weight + bias
            x_n = x_n* weight + bias
        return x_n

class InstanceAwareBatchNorm2d(nn.Module):
    def __init__(
        self, num_channels, k=3.0, eps=1e-5, momentum=0.1, threshold=1, affine=True
    ):
        super(InstanceAwareBatchNorm2d, self).__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.k = k
        self.threshold = threshold
        self.affine = affine
        self._bn = nn.BatchNorm2d(
            num_channels, eps=eps, momentum=momentum, affine=affine
        )

    def _softshrink(self, x, lbd):
        x_p = F.relu(x - lbd, inplace=True)
        x_n = F.relu(-(x + lbd), inplace=True)
        y = x_p - x_n
        return y

    def forward(self, x):
        b, c, h, w = x.size()
        sigma2, mu = torch.var_mean(x, dim=[2, 3], keepdim=True, unbiased=True)  # IN

        if self.training:
            _ = self._bn(x)
            sigma2_b, mu_b = torch.var_mean(
                x, dim=[0, 2, 3], keepdim=True, unbiased=True
            )
        else:
            if (
                self._bn.track_running_stats == False
                and self._bn.running_mean is None
                and self._bn.running_var is None
            ):  # use batch stats
                sigma2_b, mu_b = torch.var_mean(
                    x, dim=[0, 2, 3], keepdim=True, unbiased=True
                )
                # print("11111111111")
            else:
                mu_b = self._bn.running_mean.view(1, c, 1, 1)
                sigma2_b = self._bn.running_var.view(1, c, 1, 1)
                # print("@2222222222")
        if h * w <= self.threshold:
            mu_adj = mu_b
            sigma2_adj = sigma2_b
        else:
            s_mu = torch.sqrt((sigma2_b + self.eps) / (h * w))
            s_sigma2 = (sigma2_b + self.eps) * np.sqrt(2 / (h * w - 1))

            mu_adj = mu_b + self._softshrink(mu - mu_b, self.k * s_mu)

            sigma2_adj = sigma2_b + self._softshrink(
                sigma2 - sigma2_b, self.k * s_sigma2
            )
            # print(f"[INFO] sigma2_adj: {sigma2_adj.shape}")
            # print(f"[INFO] sigma2_b: {sigma2_b.shape}")
            # print(f"[INFO] sigma2 - sigma2_b: {(sigma2 - sigma2_b).shape}")

            sigma2_adj = F.relu(sigma2_adj)  # non negative

        x_n = (x - mu_adj) * torch.rsqrt(sigma2_adj + self.eps)
        if self.affine:
            weight = self._bn.weight.view(c, 1, 1)
            bias = self._bn.bias.view(c, 1, 1)
            x_n = x_n * weight + bias
        return x_n


class InstanceAwareBatchNorm1d(nn.Module):
    def __init__(
        self, num_channels, k=3.0, eps=1e-5, momentum=0.1, threshold=1, affine=True
    ):
        super(InstanceAwareBatchNorm1d, self).__init__()
        self.num_channels = num_channels
        self.k = k
        self.eps = eps
        self.threshold = threshold
        self.affine = affine
        self._bn = nn.BatchNorm1d(
            num_channels, eps=eps, momentum=momentum, affine=affine
        )

    def _softshrink(self, x, lbd):
        x_p = F.relu(x - lbd, inplace=True)
        x_n = F.relu(-(x + lbd), inplace=True)
        y = x_p - x_n
        return y

    def forward(self, x):
        b, c, l = x.size()
        sigma2, mu = torch.var_mean(x, dim=[2], keepdim=True, unbiased=True)
        if self.training:
            _ = self._bn(x)
            sigma2_b, mu_b = torch.var_mean(x, dim=[0, 2], keepdim=True, unbiased=True)
        else:
            if (
                self._bn.track_running_stats == False
                and self._bn.running_mean is None
                and self._bn.running_var is None
            ):  # use batch stats
                sigma2_b, mu_b = torch.var_mean(
                    x, dim=[0, 2], keepdim=True, unbiased=True
                )
            else:
                mu_b = self._bn.running_mean.view(1, c, 1)
                sigma2_b = self._bn.running_var.view(1, c, 1)

        if l <= self.threshold:
            mu_adj = mu_b
            sigma2_adj = sigma2_b

        else:
            s_mu = torch.sqrt((sigma2_b + self.eps) / l)  ##
            s_sigma2 = (sigma2_b + self.eps) * np.sqrt(2 / (l - 1))

            mu_adj = mu_b + self._softshrink(mu - mu_b, self.k * s_mu)
            sigma2_adj = sigma2_b + self._softshrink(
                sigma2 - sigma2_b, self.k * s_sigma2
            )
            sigma2_adj = F.relu(sigma2_adj)

        x_n = (x - mu_adj) * torch.rsqrt(sigma2_adj + self.eps)

        if self.affine:
            weight = self._bn.weight.view(c, 1)
            bias = self._bn.bias.view(c, 1)
            x_n = x_n * weight + bias

        return x_n 