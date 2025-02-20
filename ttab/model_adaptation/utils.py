# -*- coding: utf-8 -*-
import copy
import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from ttab.loads.models.resnet import (
    ResNetCifar,
    ResNetImagenet,
    ResNetMNIST,
    ViewFlatten,
)
from ttab.loads.models.wideresnet import WideResNet
from copy import deepcopy
from ttab.model_adaptation.finch import FINCH
"""optimization dynamics"""

class CustomBatchNorm(nn.Module):
    def __init__(
        self, num_channels, k=3.0, eps=1e-5, momentum=0.1, threshold=1, affine=True
    ):
        super(CustomBatchNorm, self).__init__()
        # Initialize parameters
        self.eps = eps
        self.momentum = momentum
        self._bn = nn.BatchNorm2d(
            num_channels, eps=eps, momentum=momentum, affine=affine
        )
        self.domain_difference_sum = 0
        self.k = 4
        self.prior = 0.6
        self.source_mu = None
        self.source_sigma2 = None
            
    def _softshrink(self, x, lbd):
        x_p = F.relu(x - lbd, inplace=True)
        x_n = F.relu(-(x + lbd), inplace=True)
        y = x_p - x_n
        return y

    # def forward(self, x):
    def forward(self, x, judge):
        rate = 0.2
        rate2 = 1-rate
        b, c, h, w = x.size()
        # judge='STBN'
        sigma2, mu = torch.var_mean(x, dim=[2, 3], keepdim=True, unbiased=True)  # IN
        sigma2_b = self._bn.running_var.view(1, c, 1, 1)
        s_sigma2 = (sigma2_b + self.eps) * np.sqrt(2 / (h * w - 1))

        adj = self._softshrink(sigma2 - sigma2_b, self.k * s_sigma2)
        adj_l2 = torch.linalg.norm(adj.flatten(), ord=2)

        self.domain_difference_sum += adj_l2.item()

        if judge == "iabn" and self.training is False:
            mu_b = self._bn.running_mean.view(1, c, 1, 1)
            s_mu = torch.sqrt((sigma2_b + self.eps) / (h * w))
            mu_adj = rate*(mu_b + self._softshrink(mu - mu_b, self.k * s_mu))+rate2*self.running_mean.view(1, c, 1, 1)

            sigma2_adj = rate*(sigma2_b + self._softshrink(
                sigma2 - sigma2_b, self.k * s_sigma2
            ))+rate2*self.running_var.view(1, c, 1, 1)
            # sigma2_adj = F.relu(sigma2_adj)  # non negative
            x_normalized = (x - mu_adj) * torch.rsqrt(sigma2_adj + self.eps)

        else:  # STBN
            var, mean = torch.var_mean(x, dim=[0, 2, 3], keepdim=True, unbiased=True)
            running_mean = (
                self.prior * self._bn.running_mean.view(1, c, 1, 1)
                + (1 - self.prior) * mean
            )
            running_var = (
                self.prior * self._bn.running_var.view(1, c, 1, 1)
                + (1 - self.prior) * var
            )
            x_normalized = (x - running_mean) * torch.rsqrt(running_var + self.eps)

        out = self._bn.weight.view(1, -1, 1, 1) * x_normalized + self._bn.bias.view(
            1, -1, 1, 1
        )
        return out

    def get_domain_difference_sum(self):
        sum = self.domain_difference_sum
        self.domain_difference_sum = 0
        return sum

    def sum_angle(self, x):
        b, c, h, w = x.size()

        source_mu = self._bn.running_mean
        source_sigma2 = self._bn.running_var
        sigma2_i, mu_i = torch.var_mean(x, dim=[2, 3], keepdim=True)  # IN
        sigma2_b, mu_b = torch.var_mean(x, dim=[0, 2, 3], keepdim=True)

        self.degree_all = 0
        self.sinList = []
        self.cosList = []
        self.distance = 0
        self.Dst = torch.norm(source_mu - mu_b.view(c))
        Dst = self.Dst

        for i in mu_i:
            dsi = torch.norm(i.view(c) - source_mu)
            dti = torch.norm(i.view(c) - mu_b.view(c))

            cos_angle = (dsi**2 + Dst**2 - dti**2) / (2 * dsi * Dst)
            cos = ((cos_angle) * 180 / math.pi) * dsi
            self.cosList.append(cos.item())
            # self.distance += cos.item()
            # self.distance += (sin+cos).item()
        list_size = b
        lower_index = int(list_size * 0)
        upper_index = int(list_size * 0.15)
        cos_list, _ = torch.sort(torch.tensor(self.cosList))
        cos_diff = (
            cos_list[-upper_index:].mean() - cos_list[lower_index:upper_index].mean()
        ).item()
        self.distance = cos_diff

        magic_value = 320
        """
            cifar10: 310
            cifar100: 320
            "imagenet": 4000
        """
        if self.distance > magic_value:
            return "iabn"  # iabn
        else:
            return "iabn"
        # return "stbn"    
        # self.degreeList = torch.tensor(self.degreeList)

        # degreeList, _ = torch.sort(self.degreeList)

        # self.distance = (degreeList[-3:].mean()-degreeList[:3].mean()).item()
        # print(self.distance)


def distance(a, b, c, d):
    return math.sqrt((a - c) ** 2 + (b - d) ** 2)

class DynamicBatchNorm(nn.Module):
    def __init__(self, num_features, k=3.0, eps=1e-5, momentum=0.1, threshold=1, affine=True):
        super(DynamicBatchNorm, self).__init__()
        
        # Shared parameters
        self.eps = eps
        self.momentum = momentum
        self.k = k
        self.threshold = threshold
        self.source_mu = None
        self.source_sigma2 = None

        # Separate BatchNorm layers for Homo and Heter
        self._bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=affine)
        
        # HomoDynamicBatchNorm-specific components
        self.homo_prior = nn.Parameter(torch.tensor(0.6))

        self.mode = "homo"  # "homo" for HomoDynamicBatchNorm, "heter" for HeterDynamicBatchNorm

    def set_mode(self, mode):
        """Set the mode to either 'homo' or 'heter'."""
        if mode not in ["homo", "heter"]:
            raise ValueError("Mode must be 'homo' or 'heter'")
        self.mode = mode
        # if mode == "homo":
        #     self.homo_prior = nn.Parameter(torch.tensor(0.6))
        # elif mode == "heter":
        #     self.homo_prior = nn.Parameter(torch.tensor(0.85))
        # else:
        #     raise ValueError("Invalid mode selected")

    def forward(self, x):
        if self.mode == "homo":
            self.homo_prior = nn.Parameter(torch.tensor(0.6))
        elif self.mode == "heter":
            self.homo_prior = nn.Parameter(torch.tensor(0.85))
        else:
            raise ValueError("Invalid mode selected")
        
        b, c, h, w = x.size()
        device = x.device
        
        # Clamp the prior
        # self.homo_prior.data = torch.clamp(self.homo_prior.data, 0, 1)
        
        # Calculate mean and variance
        var, mean = torch.var_mean(x, dim=[0, 2, 3], keepdim=True, unbiased=True)
        var.to(device)
        mean.to(device)
        
        running_mean = (
            self.homo_prior * self.source_mu.view(1, c, 1, 1).to(device) +
            (1 - self.homo_prior) * mean
        )
        running_var = (
            self.homo_prior * self.source_sigma2.view(1, c, 1, 1).to(device) +
            (1 - self.homo_prior) * var
        )
        
        # Normalize using HomoDynamicBatchNorm
        x_normalized = (x - running_mean) * torch.rsqrt(running_var + self.eps)
        out = self._bn.weight.view(1, -1, 1, 1) * x_normalized + self._bn.bias.view(1, -1, 1, 1)
        return out



# class DynamicBatchNorm(nn.Module):
#     def __init__(self, num_features, k=3.0, eps=1e-5, momentum=0.1, threshold=1, affine=True):
#         super(DynamicBatchNorm, self).__init__()
        
#         # Shared parameters
#         self.eps = eps
#         self.momentum = momentum
#         self.k = k
#         self.threshold = threshold
#         self.source_mu = None
#         self.source_sigma2 = None

#         # Separate BatchNorm layers for Homo and Heter
#         self.homo_bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=affine)
#         self.heter_bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=affine)
        
#         # HomoDynamicBatchNorm-specific components
#         self.homo_prior = nn.Parameter(torch.tensor(0.6))
        
#         # HeterDynamicBatchNorm-specific components
#         # self.heter_prior = 0.8
        
#         # Indicator to select forward method
#         self.mode = "homo"  # "homo" for HomoDynamicBatchNorm, "heter" for HeterDynamicBatchNorm

#     def set_mode(self, mode):
#         """Set the mode to either 'homo' or 'heter'."""
#         if mode not in ["homo", "heter"]:
#             raise ValueError("Mode must be 'homo' or 'heter'")
#         self.mode = mode

#     def forward(self, x):
#         if self.mode == "homo":
#             return self._forward_homo(x)
#         elif self.mode == "heter":
#             return self._forward_heter(x)
#         else:
#             raise ValueError("Invalid mode selected")

#     def _forward_homo(self, x):
#         """HomoDynamicBatchNorm forward method."""
#         b, c, h, w = x.size()
#         device = x.device
        
#         # Clamp the prior
#         self.homo_prior.data = torch.clamp(self.homo_prior.data, 0, 1)
        
#         # Calculate mean and variance
#         var, mean = torch.var_mean(x, dim=[0, 2, 3], keepdim=True, unbiased=True)
#         var.to(device)
#         mean.to(device)
        
#         running_mean = (
#             self.homo_prior * self.source_mu.view(1, c, 1, 1).to(device) +
#             (1 - self.homo_prior) * mean
#         )
#         running_var = (
#             self.homo_prior * self.source_sigma2.view(1, c, 1, 1).to(device) +
#             (1 - self.homo_prior) * var
#         )
        
#         # Normalize using HomoDynamicBatchNorm
#         x_normalized = (x - running_mean) * torch.rsqrt(running_var + self.eps)
#         out = self.homo_bn.weight.view(1, -1, 1, 1) * x_normalized + self.homo_bn.bias.view(1, -1, 1, 1)
#         return out

#     def _forward_heter(self, x):
#         """HeterDynamicBatchNorm forward method."""
#         rate = 0.2
#         rate2 = 1 - rate
#         b, c, h, w = x.size()
#         device = x.device
        
#         # Calculate mean and variance
#         sigma2, mu = torch.var_mean(x, dim=[2, 3], keepdim=True, unbiased=True)  # IN
#         sigma2_threshold, mu_threshold = torch.var_mean(x, dim=[0, 2, 3], keepdim=True, unbiased=True)
#         sigma2_b = self.heter_bn.running_var.view(1, c, 1, 1).to(device)
#         s_sigma2 = (sigma2_threshold + self.eps) * np.sqrt(2 / (h * w - 1))
        
#         # Adjustments
#         adj = self._softshrink(sigma2 - sigma2_b, self.k * s_sigma2)
#         mu_b = self.heter_bn.running_mean.view(1, c, 1, 1)
#         s_mu = torch.sqrt((sigma2_threshold + self.eps) / (h * w))
        
#         mu_adj = rate * (mu_b + self._softshrink(mu - mu_b, self.k * s_mu)) + rate2 * self.source_mu.view(1, c, 1, 1).to(device)
#         sigma2_adj = rate * (sigma2_b + self._softshrink(sigma2 - sigma2_b, self.k * s_sigma2)) + rate2 * self.source_sigma2.view(1, c, 1, 1).to(device)
#         sigma2_adj = F.relu(sigma2_adj)  # non-negative
        
#         # Normalize using HeterDynamicBatchNorm
#         x_normalized = (x - mu_adj) * torch.rsqrt(sigma2_adj + self.eps)
#         out = self.heter_bn.weight.view(1, -1, 1, 1) * x_normalized + self.heter_bn.bias.view(1, -1, 1, 1)
#         return out

#     def _softshrink(self, x, lbd):
#         """Softshrink function used in HeterDynamicBatchNorm."""
#         x_p = F.relu(x - lbd, inplace=True)
#         x_n = F.relu(-(x + lbd), inplace=True)
#         y = x_p - x_n
#         return y
class HomoDynamicBatchNorm(nn.Module):
    def __init__(
        self, num_features, k=3.0, 
        eps=1e-5, momentum=0.1, threshold=1, 
        affine=True, 
    ):
        super(HomoDynamicBatchNorm, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self._bn = nn.BatchNorm2d(
            num_features, eps=eps, momentum=momentum, affine=affine
        )
        self.domain_difference_sum = 0
        self.k = 4
        self.prior = 0.6
        self.source_mu = None
        self.source_sigma2 = None
            
    def _softshrink(self, x, lbd):
        x_p = F.relu(x - lbd, inplace=True)
        x_n = F.relu(-(x + lbd), inplace=True)
        y = x_p - x_n
        return y

    # def forward(self, x):
    def forward(self, x):
        # rate = 0.4
        # rate2 = 1-rate
        b, c, h, w = x.size()
        device = x.device  # 获取输入张量的设备
        
        sigma2, mu = torch.var_mean(x, dim=[2, 3], keepdim=True, unbiased=True)  # IN
        sigma2_b = self._bn.running_var.view(1, c, 1, 1).to(device)  # 移动到相同设备
        s_sigma2 = (sigma2_b + self.eps) * np.sqrt(2 / (h * w - 1))

        adj = self._softshrink(sigma2 - sigma2_b, self.k * s_sigma2)
        adj_l2 = torch.linalg.norm(adj.flatten(), ord=2)
        self.domain_difference_sum += adj_l2.item()

        var, mean = torch.var_mean(x, dim=[0, 2, 3], keepdim=True, unbiased=True)
        running_mean = (
            self.prior * self.source_mu.view(1, c, 1, 1).to(device)  # 移动到相同设备
            + (1 - self.prior) * mean
        )
        running_var = (
            self.prior * self.source_sigma2.view(1, c, 1, 1).to(device)  # 移动到相同设备
            + (1 - self.prior) * var
        )
        x_normalized = (x - running_mean) * torch.rsqrt(running_var + self.eps)

        out = self._bn.weight.view(1, -1, 1, 1).to(device) * x_normalized + self._bn.bias.view(
            1, -1, 1, 1
        ).to(device)  # 移动到相同设备
        return out
    def get_domain_difference_sum(self):
        sum = self.domain_difference_sum
        self.domain_difference_sum = 0
        return sum

class HeterDynamicBatchNorm(nn.Module):
    def __init__(
        self, num_features, k=3.0, eps=1e-5, momentum=0.1, threshold=1, affine=True
    ):
        super(HeterDynamicBatchNorm, self).__init__()
        # Initialize parameters
        self.eps = eps
        self.momentum = momentum
        self._bn = nn.BatchNorm2d(
            num_features, eps=eps, momentum=momentum, affine=affine
        )
        self.domain_difference_sum = 0
        self.k = 4
        # self.prior = 0.6
        self.source_mu = None
        self.source_sigma2 = None

            
    def _softshrink(self, x, lbd):
        x_p = F.relu(x - lbd, inplace=True)
        x_n = F.relu(-(x + lbd), inplace=True)
        y = x_p - x_n
        return y

    # def forward(self, x):
    def forward(self, x):
        rate = 0.15
        rate2 = 1-rate
        b, c, h, w = x.size()
        device = x.device  # 获取输入张量的设备
        
        sigma2, mu = torch.var_mean(x, dim=[2, 3], keepdim=True, unbiased=True)  # IN
        sigma2_b = self._bn.running_var.view(1, c, 1, 1).to(device)  # 移动到相同设备
        s_sigma2 = (sigma2_b + self.eps) * np.sqrt(2 / (h * w - 1))

        adj = self._softshrink(sigma2 - sigma2_b, self.k * s_sigma2)
        adj_l2 = torch.linalg.norm(adj.flatten(), ord=2)
        self.domain_difference_sum += adj_l2.item()


        mu_b = self._bn.running_mean.view(1, c, 1, 1)
        s_mu = torch.sqrt((sigma2_b + self.eps) / (h * w))
        # mu_adj = rate *(mu_b + self._softshrink(mu - mu_b, self.k * s_mu)) + rate2*self.source_mu.view(1, c, 1, 1).to(device)  # 移动到相同设备

        # sigma2_adj = rate*(sigma2_b + self._softshrink(
        #     sigma2 - sigma2_b, self.k * s_sigma2
        # )) + rate2*self.source_sigma2.view(1, c, 1, 1).to(device)  # 移动到相同设备
        mu_adj = rate *(mu_b + self._softshrink(mu - mu_b, self.k * s_mu)) + rate2*self.source_mu.view(1, c, 1, 1).to(device)  # 移动到相同设备

        sigma2_adj = rate*(sigma2_b + self._softshrink(
            sigma2 - sigma2_b, self.k * s_sigma2
        )) + rate2*self.source_sigma2.view(1, c, 1, 1).to(device)  # 移动到相同设备
        # sigma2_adj = F.relu(sigma2_adj)  # non negative
        x_normalized = (x - mu_adj) * torch.rsqrt(sigma2_adj + self.eps)

        out = self._bn.weight.view(1, -1, 1, 1) * x_normalized + self._bn.bias.view(
            1, -1, 1, 1
        ).to(device)  # 移动到相同设备
        return out
    def get_domain_difference_sum(self):
        sum = self.domain_difference_sum
        self.domain_difference_sum = 0
        return sum
    
def define_optimizer(meta_conf, params, lr=1e-3):
    """Set up optimizer for adaptation."""
    weight_decay = meta_conf.weight_decay if hasattr(meta_conf, "weight_decay") else 0

    if not hasattr(meta_conf, "optimizer") or meta_conf.optimizer == "SGD":
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=meta_conf.momentum if hasattr(meta_conf, "momentum") else 0.9,
            dampening=meta_conf.dampening if hasattr(meta_conf, "dampening") else 0,
            weight_decay=weight_decay,
            nesterov=meta_conf.nesterov if hasattr(meta_conf, "nesterov") else True,
        )
    elif meta_conf.optimizer == "Adam":
        return torch.optim.Adam(
            params,
            lr=lr,
            betas=(meta_conf.beta if hasattr(meta_conf, "beta") else 0.9, 0.999),
            weight_decay=weight_decay,
        )
    else:
        raise NotImplementedError


def lr_scheduler(optimizer, iter_ratio, gamma=10, power=0.75):
    decay = (1 + gamma * iter_ratio) ** (-power)
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["lr0"] * decay
    return optimizer


class SAM(torch.optim.Optimizer):
    """
    SAM is an optimizer proposed to seek parameters that lie in neighborhoods having uniformly low loss.

    Sharpness-Aware Minimization for Efficiently Improving Generalization
    https://arxiv.org/abs/2010.01412
    """

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (
                    (torch.pow(p, 2) if group["adaptive"] else 1.0)
                    * p.grad
                    * scale.to(p)
                )
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert (
            closure is not None
        ), "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(
            closure
        )  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][
            0
        ].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack(
                [
                    ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad)
                    .norm(p=2)
                    .to(shared_device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p=2,
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


"""method-wise modification on model structure"""

# for bn_adapt
def modified_bn_forward(self, input):
    """
    Leverage the statistics already computed on the seen data as a prior and infer the test statistics for each test batch as a weighted sum of
    prior statistics and estimated statistics on the current batch.

    Improving robustness against common corruptions by covariate shift adaptation
    https://arxiv.org/abs/2006.16971
    """
    est_mean = torch.zeros(self.running_mean.shape, device=self.running_mean.device)
    est_var = torch.ones(self.running_var.shape, device=self.running_var.device)
    nn.functional.batch_norm(input, est_mean, est_var, None, None, True, 1.0, self.eps)
    running_mean = self.prior * self.running_mean + (1 - self.prior) * est_mean
    running_var = self.prior * self.running_var + (1 - self.prior) * est_var
    return nn.functional.batch_norm(
        input, running_mean, running_var, self.weight, self.bias, False, 0, self.eps
    )


class shared_ext_from_layer4(nn.Module):
    """
    Select all layers before layer4 and layer4 as the shared feature extractor for the main and auxiliary branches.

    Only used for ResNets.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.layers = self._select_layers()

    def forward(self, x):
        return self.model.forward_features(x)

    def _select_layers(self):
        if isinstance(self.model, ResNetImagenet):
            return {
                "conv1": self.model.conv1,
                "bn1": self.model.bn1,
                "relu": self.model.relu,
                "maxpool": self.model.maxpool,
                "layer1": self.model.layer1,
                "layer2": self.model.layer2,
                "layer3": self.model.layer3,
                "layer4": self.model.layer4,
                "avgpool": self.model.avgpool,
                "ViewFlatten": ViewFlatten(),
            }
        elif isinstance(self.model, ResNetMNIST):
            return {
                "conv1": self.model.conv1,
                "bn1": self.model.bn1,
                "relu": self.model.relu,
                "maxpool": self.model.maxpool,
                "layer1": self.model.layer1,
                "layer2": self.model.layer2,
                "layer3": self.model.layer3,
                "layer4": self.model.layer4,
                "ViewFlatten": ViewFlatten(),
            }
        else:
            raise NotImplementedError

    def make_train(self):
        for _, layer_module in self.layers.items():
            layer_module.train()

    def make_eval(self):
        for _, layer_module in self.layers.items():
            layer_module.eval()


class shared_ext_from_layer3(nn.Module):
    """
    Select all layers before layer3 and layer3 as the shared feature extractor for the main and auxiliary branches.

    Only used for ResNets.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.layers = self._select_layers()

    def forward(self, x):
        return self.model.forward_features(x)

    def _select_layers(self):
        if isinstance(self.model, ResNetCifar):
            return {
                "conv1": self.model.conv1,
                "bn1": self.model.bn1,
                "relu": self.model.relu,
                "layer1": self.model.layer1,
                "layer2": self.model.layer2,
                "layer3": self.model.layer3,
                "avgpool": self.model.avgpool,
                "ViewFlatten": ViewFlatten(),
            }
        elif isinstance(self.model, (ResNetImagenet, ResNetMNIST)):
            return {
                "conv1": self.model.conv1,
                "bn1": self.model.bn1,
                "relu": self.model.relu,
                "maxpool": self.model.maxpool,
                "layer1": self.model.layer1,
                "layer2": self.model.layer2,
                "layer3": self.model.layer3,
            }
        elif isinstance(self.model, WideResNet):
            return {
                "conv1": self.model.conv1,
                "layer1": self.model.layer1,
                "layer2": self.model.layer2,
                "layer3": self.model.layer3,
                "bn1": self.model.bn1,
                "relu": self.model.relu,
                "avgpool": self.model.avgpool,
                "ViewFlatten": ViewFlatten(),
            }
        else:
            raise NotImplementedError

    def make_train(self):
        for _, layer_module in self.layers.items():
            layer_module.train()

    def make_eval(self):
        for _, layer_module in self.layers.items():
            layer_module.eval()


class shared_ext_from_layer2(nn.Module):
    """
    Select all layers before layer2 and layer2 as the shared feature extractor for the main and auxiliary branches.

    Only used for ResNets.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.layers = self._select_layers()

    def forward(self, x):
        return self.model.forward_features(x)

    def _select_layers(self):
        if isinstance(self.model, ResNetCifar):
            return {
                "conv1": self.model.conv1,
                "bn1": self.model.bn1,
                "relu": self.model.relu,
                "layer1": self.model.layer1,
                "layer2": self.model.layer2,
            }
        elif isinstance(self.model, (ResNetImagenet, ResNetMNIST)):
            return {
                "conv1": self.model.conv1,
                "bn1": self.model.bn1,
                "relu": self.model.relu,
                "maxpool": self.model.maxpool,
                "layer1": self.model.layer1,
                "layer2": self.model.layer2,
            }
        elif isinstance(self.model, WideResNet):
            return {
                "conv1": self.model.conv1,
                "layer1": self.model.layer1,
                "layer2": self.model.layer2,
            }
        else:
            raise NotImplementedError

    def make_train(self):
        for _, layer_module in self.layers.items():
            layer_module.train()

    def make_eval(self):
        for _, layer_module in self.layers.items():
            layer_module.eval()


def head_from_classifier(model, dim_out):
    """Select the last classifier layer in ResNets as head."""
    # Self-supervised task used in TTT is rotation prediction. Thus the out_features = 4.
    head = nn.Linear(
        in_features=model.classifier.in_features, out_features=dim_out, bias=True
    )
    return head


def head_from_last_layer1(model, dim_out):
    """
    Select the layer 3 or 4 and the following classifier layer as head.

    Only used for ResNets.
    """
    if isinstance(model, ResNetCifar):
        head = copy.deepcopy([model.layer3, model.avgpool])
        head.append(ViewFlatten())
        head.append(nn.Linear(model.classifier.in_features, dim_out, bias=False))
    elif isinstance(model, ResNetImagenet):
        head = copy.deepcopy([model.layer4, model.avgpool])
        head.append(ViewFlatten())
        head.append(nn.Linear(model.classifier.in_features, dim_out, bias=False))
    elif isinstance(model, ResNetMNIST):
        head = copy.deepcopy([model.layer4])
        head.append(ViewFlatten())
        head.append(nn.Linear(model.classifier.in_features, dim_out, bias=False))
    elif isinstance(model, WideResNet):
        head = copy.deepcopy([model.layer3, model.bn1, model.relu, model.avgpool])
        head.append(ViewFlatten())
        head.append(nn.Linear(model.classifier.in_features, dim_out, bias=False))
    elif isinstance(model, models.ResNet):
        # for torchvision.models.resnet50
        head = copy.deepcopy([model.layer4, model.avgpool])
        head.append(ViewFlatten())
        head.append(nn.Linear(model.fc.in_features, dim_out, bias=False))
    else:
        raise NotImplementedError

    return nn.Sequential(*head)


class ExtractorHead(nn.Module):
    """
    Combine the extractor and the head together in ResNets.
    """

    def __init__(self, ext, head):
        super(ExtractorHead, self).__init__()
        self.ext = ext
        self.head = head

    def forward(self, x):
        return self.head(self.ext(x))

    def make_train(self):
        self.ext.make_train()
        self.head.train()

    def make_eval(self):
        self.ext.make_eval()
        self.head.eval()


class VitExtractor(nn.Module):
    """
    Combine the extractor and the head together in ViTs.
    """

    def __init__(self, model):
        super(VitExtractor, self).__init__()
        self.model = model
        self.layers = self._select_layers()

    def forward(self, x):
        x = self.model.forward_features(x)
        if self.model.global_pool:
            x = (
                x[:, self.model.num_prefix_tokens :].mean(dim=1)
                if self.model.global_pool == "avg"
                else x[:, 0]
            )
        x = self.model.fc_norm(x)
        return x

    def _select_layers(self):
        layers = []
        for named_module, module in self.model.named_children():
            if not module == self.model.get_classifier():
                layers.append(module)
        return layers

    def make_train(self):
        for layer in self.layers:
            layer.train()

    def make_eval(self):
        for layer in self.layers:
            layer.eval()


# for ttt++
class FeatureQueue:
    def __init__(self, dim, length):
        self.length = length
        self.queue = torch.zeros(length, dim)
        self.ptr = 0

    @torch.no_grad()
    def update(self, feat):

        batch_size = feat.shape[0]
        assert self.length % batch_size == 0  # for simplicity

        # replace the features at ptr (dequeue and enqueue)
        self.queue[self.ptr : self.ptr + batch_size] = feat
        self.ptr = (self.ptr + batch_size) % self.length  # move pointer

    def get(self):
        cnt = (self.queue[-1] != 0).sum()
        if cnt.item():
            return self.queue
        else:
            return None


# for note
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
            else:
                mu_b = self._bn.running_mean.view(1, c, 1, 1)
                sigma2_b = self._bn.running_var.view(1, c, 1, 1)

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
        self.k = 1
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
        b, l = x.size()  # 现在是2维: batch和feature maps
        sigma2, mu = torch.var_mean(x, dim=[1], keepdim=True, unbiased=True)  # 改为dim=[1]
        if self.training:
            _ = self._bn(x)
            sigma2_b, mu_b = torch.var_mean(x, dim=[0, 1], keepdim=True, unbiased=True)  # 改为dim=[0,1]
        else:
            if (
                self._bn.track_running_stats == False
                and self._bn.running_mean is None
                and self._bn.running_var is None
            ):  # use batch stats
                sigma2_b, mu_b = torch.var_mean(
                    x, dim=[0, 1], keepdim=True, unbiased=True  # 改为dim=[0,1]
                )
            else:
                print("[info] x size :", x.size())
                mu_b = self._bn.running_mean.view(1,l)      # 改为(1,1)
                print("[info] mu_b size :", mu_b.size())
                sigma2_b = self._bn.running_var.view(1,l)   # 改为(1,1)

        if l <= self.threshold:
            mu_adj = mu_b
            sigma2_adj = sigma2_b

        else:
            s_mu = torch.sqrt((sigma2_b + self.eps) / l)
            s_sigma2 = (sigma2_b + self.eps) * np.sqrt(2 / (l - 1))

            mu_adj = mu_b + self._softshrink(mu - mu_b, self.k * s_mu)
            sigma2_adj = sigma2_b + self._softshrink(
                sigma2 - sigma2_b, self.k * s_sigma2
            )
            sigma2_adj = F.relu(sigma2_adj)

        x_n = (x - mu_adj) * torch.rsqrt(sigma2_adj + self.eps)

        if self.affine:
            weight = self._bn.weight.view(1,l)  # 改为(1,1)
            bias = self._bn.bias.view(1,l)      # 改为(1,1)
            x_n = x_n * weight + bias

        return x_n
    # def forward(self, x):
    #     print("[info] x size :", x.size())
    #     b, c, l = x.size()
    #     sigma2, mu = torch.var_mean(x, dim=[2], keepdim=True, unbiased=True)
    #     if self.training:
    #         _ = self._bn(x)
    #         sigma2_b, mu_b = torch.var_mean(x, dim=[0, 2], keepdim=True, unbiased=True)
    #     else:
    #         if (
    #             self._bn.track_running_stats == False
    #             and self._bn.running_mean is None
    #             and self._bn.running_var is None
    #         ):  # use batch stats
    #             sigma2_b, mu_b = torch.var_mean(
    #                 x, dim=[0, 2], keepdim=True, unbiased=True
    #             )
    #         else:
    #             mu_b = self._bn.running_mean.view(1, c, 1)
    #             sigma2_b = self._bn.running_var.view(1, c, 1)

    #     if l <= self.threshold:
    #         mu_adj = mu_b
    #         sigma2_adj = sigma2_b

    #     else:
    #         s_mu = torch.sqrt((sigma2_b + self.eps) / l)  ##
    #         s_sigma2 = (sigma2_b + self.eps) * np.sqrt(2 / (l - 1))

    #         mu_adj = mu_b + self._softshrink(mu - mu_b, self.k * s_mu)
    #         sigma2_adj = sigma2_b + self._softshrink(
    #             sigma2 - sigma2_b, self.k * s_sigma2
    #         )
    #         sigma2_adj = F.relu(sigma2_adj)

    #     x_n = (x - mu_adj) * torch.rsqrt(sigma2_adj + self.eps)

    #     if self.affine:
    #         weight = self._bn.weight.view(c, 1)
    #         bias = self._bn.bias.view(c, 1)
    #         x_n = x_n * weight + bias

    #     return x_n


# for rotta
class MomentumBN(nn.Module):
    def __init__(self, bn_layer: nn.BatchNorm2d, momentum):
        super().__init__()
        self.num_features = bn_layer.num_features
        self.momentum = momentum
        if (
            bn_layer.track_running_stats
            and bn_layer.running_var is not None
            and bn_layer.running_mean is not None
        ):
            self.register_buffer("source_mean", deepcopy(bn_layer.running_mean))
            self.register_buffer("source_var", deepcopy(bn_layer.running_var))
            self.source_num = bn_layer.num_batches_tracked
        self.weight = deepcopy(bn_layer.weight)
        self.bias = deepcopy(bn_layer.bias)

        self.register_buffer("target_mean", torch.zeros_like(self.source_mean))
        self.register_buffer("target_var", torch.ones_like(self.source_var))
        self.eps = bn_layer.eps

        self.current_mu = None
        self.current_sigma = None

    def forward(self, x):
        raise NotImplementedError


class RobustBN1d(MomentumBN):
    def forward(self, x):
        if self.training:
            b_var, b_mean = torch.var_mean(
                x, dim=0, unbiased=False, keepdim=False
            )  # (C,)
            mean = (1 - self.momentum) * self.source_mean + self.momentum * b_mean
            var = (1 - self.momentum) * self.source_var + self.momentum * b_var
            self.source_mean, self.source_var = deepcopy(mean.detach()), deepcopy(
                var.detach()
            )
            mean, var = mean.view(1, -1), var.view(1, -1)
        else:
            mean, var = self.source_mean.view(1, -1), self.source_var.view(1, -1)

        x = (x - mean) / torch.sqrt(var + self.eps)
        weight = self.weight.view(1, -1)
        bias = self.bias.view(1, -1)

        return x * weight + bias


class RobustBN2d(MomentumBN):
    def forward(self, x):
        if self.training:
            b_var, b_mean = torch.var_mean(
                x, dim=[0, 2, 3], unbiased=False, keepdim=False
            )  # (C,)
            mean = (1 - self.momentum) * self.source_mean + self.momentum * b_mean
            var = (1 - self.momentum) * self.source_var + self.momentum * b_var
            self.source_mean, self.source_var = copy.deepcopy(mean.detach()), deepcopy(
                var.detach()
            )
            mean, var = mean.view(1, -1, 1, 1), var.view(1, -1, 1, 1)
        else:
            mean, var = self.source_mean.view(1, -1, 1, 1), self.source_var.view(
                1, -1, 1, 1
            )

        x = (x - mean) / torch.sqrt(var + self.eps)
        weight = self.weight.view(1, -1, 1, 1)
        bias = self.bias.view(1, -1, 1, 1)

        return x * weight + bias


"""Auxiliary tasks"""

# rotation prediction task
def tensor_rot_90(x):
    return x.flip(2).transpose(1, 2)


def tensor_rot_180(x):
    return x.flip(2).flip(1)


def tensor_rot_270(x):
    return x.transpose(1, 2).flip(2)


def rotate_batch_with_labels(batch, labels):
    images = []
    for img, label in zip(batch, labels):
        if label == 1:
            img = tensor_rot_90(img)
        elif label == 2:
            img = tensor_rot_180(img)
        elif label == 3:
            img = tensor_rot_270(img)
        images.append(img.unsqueeze(0))
    return torch.cat(images)


def rotate_batch(batch, label, device, generator=None):
    if label == "rand":
        labels = torch.randint(
            4, (len(batch),), generator=generator, dtype=torch.long
        ).to(device)
    elif label == "expand":
        labels = torch.cat(
            [
                torch.zeros(len(batch), dtype=torch.long),
                torch.zeros(len(batch), dtype=torch.long) + 1,
                torch.zeros(len(batch), dtype=torch.long) + 2,
                torch.zeros(len(batch), dtype=torch.long) + 3,
            ]
        ).to(device)
        batch = batch.repeat((4, 1, 1, 1))

    return rotate_batch_with_labels(batch, labels), labels


"""loss-related functions."""


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def teacher_student_softmax_entropy(
    x: torch.Tensor, x_ema: torch.Tensor
) -> torch.Tensor:
    """Cross entropy between the teacher and student predictions."""
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)


def marginal_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1), avg_logits


def entropy(input):
    bs = input.size(0)
    ent = -input * torch.log(input + 1e-5)
    ent = torch.sum(ent, dim=1)
    return ent


def covariance(features):
    assert len(features.size()) == 2, "TODO: multi-dimensional feature map covariance"
    n = features.shape[0]
    tmp = torch.ones((1, n), device=features.device) @ features
    cov = (features.t() @ features - (tmp.t() @ tmp) / n) / (n - 1)
    return cov


def coral(cs, ct):
    d = cs.shape[0]
    loss = (cs - ct).pow(2).sum() / (4.0 * d**2)
    return loss


def linear_mmd(ms, mt):
    loss = (ms - mt).pow(2).mean()
    return loss


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, device, epsilon=0.1, reduction=True):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.device = device

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(
            1, targets.unsqueeze(1).cpu(), 1
        )
        targets = targets.to(self.device)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).sum(dim=1)
        if self.reduction:
            return loss.mean()
        else:
            return loss


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class HLoss(nn.Module):
    def __init__(self, temp_factor=1.0):
        super().__init__()
        self.temp_factor = temp_factor

    def forward(self, x):

        softmax = F.softmax(x / self.temp_factor, dim=1)
        entropy = -softmax * torch.log(softmax + 1e-6)
        b = entropy.mean()

        return b

"""
for vida
"""
class ViDAInjectedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False, r=4, r2 = 64):
        super().__init__()

        self.linear_vida = nn.Linear(in_features, out_features, bias)
        self.vida_down = nn.Linear(in_features, r, bias=False)
        self.vida_up = nn.Linear(r, out_features, bias=False)
        self.vida_down2 = nn.Linear(in_features, r2, bias=False)
        self.vida_up2 = nn.Linear(r2, out_features, bias=False)
        self.scale1 = 1.0
        self.scale2 = 1.0

        nn.init.normal_(self.vida_down.weight, std=1 / r**2)
        nn.init.zeros_(self.vida_up.weight)

        nn.init.normal_(self.vida_down2.weight, std=1 / r2**2)
        nn.init.zeros_(self.vida_up2.weight)

    def forward(self, input):
        # return self.linear_vida(input) + self.vida_up(self.vida_down(input)) * self.scale1 
        return self.linear_vida(input) \
            *torch.sigmoid(self.vida_up(self.vida_down(input)) * self.scale1)
    
def inject_trainable_vida(
    model: nn.Module,
    r: int = 4,
    r2: int = 16,
):
    require_grad_params = []
    names = []

    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            weight = module.weight
            bias = module.bias
            tmp = ViDAInjectedLinear(
                module.in_features,
                module.out_features,
                module.bias is not None,
                r,
                r2,
            )
            tmp.linear_vida.weight = weight
            if bias is not None:
                tmp.linear_vida.bias = bias

            # Switch the module
            setattr(model, name, tmp)

            require_grad_params.extend(tmp.vida_up.parameters())
            require_grad_params.extend(tmp.vida_down.parameters())
            tmp.vida_up.weight.requires_grad = True
            tmp.vida_down.weight.requires_grad = True

            require_grad_params.extend(tmp.vida_up2.parameters())
            require_grad_params.extend(tmp.vida_down2.parameters())
            tmp.vida_up2.weight.requires_grad = True
            tmp.vida_down2.weight.requires_grad = True

            names.append(name)   
    return require_grad_params, names


"""
for dyn
"""
# class ClusterAwareBatchNorm2d(nn.Module):
#     def __init__(self, num_channels, eps=1e-5, momentum=0.1, affine=True):
#         super(ClusterAwareBatchNorm2d, self).__init__()
#         self.mode = "cluster"
#         self.num_channels = num_channels
#         self.eps = eps
#         self.affine = affine
#         # self.k = 3
#         self._bn = nn.BatchNorm2d(
#             num_channels, eps=eps, momentum=momentum, affine=affine
#         )
#         self.source_rate = nn.parameter.Parameter(torch.tensor(0.8))
#     def Lisc(self, x, mu_b, sigma2_b, source_rate):
#         b, c, _, _ = x.size()
#         rate_ = 1 - source_rate

#         # 计算均值和方差
#         sigma2, mu = torch.var_mean(x, dim=[2, 3], keepdim=True, unbiased=True)
#         mu_flat = mu.view(b, c)
#         sigma2_flat = sigma2.view(b, c)

#         # 获取聚类标签
#         data = mu_flat
#         c1, _ = FINCH(data)
#         unique_labels, inverse_indices = torch.unique(c1, dim=0, return_inverse=True)

#         # 将标签转换为one-hot编码
#         label_onehot = nn.functional.one_hot(inverse_indices, num_classes=len(unique_labels)).float().to(x.device)
#         # 聚类均值和方差
#         label_sums = torch.matmul(label_onehot.transpose(0, 1), mu_flat)
#         label_counts = label_onehot.sum(dim=0, keepdim=True).transpose(0, 1) + self.eps
#         cluster_mus = label_sums / label_counts

#         label_vars = torch.matmul(label_onehot.transpose(0, 1), sigma2_flat)
#         cluster_sigma2s = label_vars / label_counts
        
#         mu_diff_squared = (mu_flat.unsqueeze(1) - cluster_mus.unsqueeze(0))**2
#         weighted_mu_diff_squared = torch.einsum('blc,bl->lc', mu_diff_squared, label_onehot) / label_counts
#         # mu_diff_squared = (mu_flat.unsqueeze(1) - cluster_mus.unsqueeze(0)) ** 2
#         # weighted_mu_diff_squared = torch.matmul(label_onehot.transpose(0, 1), mu_diff_squared) / label_counts
#         cluster_sigma2s += weighted_mu_diff_squared
        
#         # 计算用于标准化的均值和方差
#         cluster_mus_expanded = torch.matmul(label_onehot, cluster_mus).view(b, c, 1, 1)
#         cluster_sigma2s_expanded = torch.matmul(label_onehot, cluster_sigma2s).view(b, c, 1, 1)

#         # 扩展 mu_b 和 sigma2_b
#         mu_b_expanded = mu_b.expand(b, -1, -1, -1)
#         sigma2_b_expanded = sigma2_b.expand(b, -1, -1, -1)

#         # 标准化
#         x = (x - (rate_ * cluster_mus_expanded + source_rate * mu_b_expanded)) * torch.rsqrt(
#             (rate_ * cluster_sigma2s_expanded + source_rate * sigma2_b_expanded) + self.eps
#         )

#         return x
   
#     def forward(self, x):
#         print(self.mode)
#         b, c, h, w = x.size()
#         mu_s = self._bn.running_mean.view(1, c, 1, 1)
#         sigma2_s = self._bn.running_var.view(1, c, 1, 1)

#         x_n = self.Lisc(x, mu_s, sigma2_s, self.source_rate)  # + self.eps
#         weight = self._bn.weight.view(c, 1, 1)
#         bias = self._bn.bias.view(c, 1, 1)
#         x_n = x_n * weight + bias
#         return x_n
    
# class ClusterAwareBatchNorm1d(nn.Module):
#     def __init__(self, num_channels , eps=1e-5, momentum=0.1, affine=True):
#         super(ClusterAwareBatchNorm1d, self).__init__()
#         self.mode = "cluster"
#         self.num_channels = num_channels 
#         self.eps = eps
#         self.affine = affine
#         # BatchNorm1d for normalizing over feature channels
#         self._bn = nn.BatchNorm1d(num_channels , eps=eps, momentum=momentum, affine=affine)
#         self.source_rate = nn.Parameter(torch.tensor(0.8))  # learnable source rate

#     def Lisc(self, x, mu_b, sigma2_b, source_rate):
#         b, c = x.size()  # (batch_size, num_features)

#         rate_ = 1 - source_rate
     

#         # Compute the mean and variance across the batch dimension
#         sigma2, mu = torch.var_mean(x, dim=1, unbiased=True)
        

#         # Flatten to use clustering
#         mu_flat = mu.view(b,1)
#         sigma2_flat = sigma2.view(b,1)

#         # Apply clustering (assuming FINCH is implemented elsewhere)
#         data = mu_flat
#         c1, _ = FINCH(data)  # Assuming FINCH returns cluster labels
#         unique_labels, inverse_indices = torch.unique(c1, return_inverse=True)

#         # One-hot encoding of cluster labels
#         label_onehot = F.one_hot(inverse_indices, num_classes=len(unique_labels)).float().to(x.device)

#         # Cluster statistics
#         label_sums = torch.matmul(label_onehot.transpose(0, 1), mu_flat)
#         label_counts = label_onehot.sum(dim=0, keepdim=True).transpose(0, 1) + self.eps
#         cluster_mus = label_sums / label_counts

#         label_vars = torch.matmul(label_onehot.transpose(0, 1), sigma2_flat)
#         cluster_sigma2s = label_vars / label_counts

#         # Adjust variance using the cluster-aware statistics
#         mu_diff_squared = (mu_flat.unsqueeze(1) - cluster_mus.unsqueeze(0))**2
#         weighted_mu_diff_squared = torch.einsum('blc,bl->lc', mu_diff_squared, label_onehot) / label_counts
#         cluster_sigma2s += weighted_mu_diff_squared

#         # Expand cluster statistics to match batch dimensions
#         cluster_mus_expanded = torch.matmul(label_onehot, cluster_mus).view(b, 1)
#         cluster_sigma2s_expanded = torch.matmul(label_onehot, cluster_sigma2s).view(b, 1)

#         # Expand mu_b and sigma2_b
#         mu_b_expanded = mu_b.expand(b, -1)
#         sigma2_b_expanded = sigma2_b.expand(b, -1)

#         # Normalize using cluster-aware stats
#         x = (x - (rate_ * cluster_mus_expanded + source_rate * mu_b_expanded)) * torch.rsqrt(
#             (rate_ * cluster_sigma2s_expanded + source_rate * sigma2_b_expanded) + self.eps
#         )

#         return x

#     def forward(self, x):
#         print(self.mode)
#         b, c = x.size()  # (batch_size, num_features)

#         mu_s = self._bn.running_mean.view(1, c)
#         sigma2_s = self._bn.running_var.view(1, c)

#         # Apply the cluster-aware normalization
#         x_n = self.Lisc(x, mu_s, sigma2_s, self.source_rate)
        
#         # Scale and shift with learned affine parameters (if affine=True)
#         weight = self._bn.weight.view(1, c)
#         bias = self._bn.bias.view(1, c)
        
#         # Apply affine transformation
#         x_n = x_n * weight + bias
#         return x_n



class ClusterAwareBatchNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-5, momentum=0.1, affine=True):
        super(ClusterAwareBatchNorm2d, self).__init__()
        self.mode = 'cluster'
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        # self.k = 3
        self._bn = nn.BatchNorm2d(
            num_channels, eps=eps, momentum=momentum, affine=affine
        )
        self.source_rate = nn.parameter.Parameter(torch.tensor(0.8))
        # self.test_rate = nn.parameter.Parameter(torch.tensor(0.0))
        # self.Domain_discriminator = False
    def Lisc(self, x, mu_b, sigma2_b, source_rate):
        b, c, _, _ = x.size()
        rate_ = 1 - source_rate

        # 计算均值和方差
        sigma2, mu = torch.var_mean(x, dim=[2, 3], keepdim=True, unbiased=True)
        mu_flat = mu.view(b, c)
        sigma2_flat = sigma2.view(b, c)

        # 获取聚类标签
        data = mu_flat
        c1, _ = FINCH(data)
        unique_labels, inverse_indices = torch.unique(c1, dim=0, return_inverse=True)
        # print('[info] unique_labels length:', len(unique_labels))
        
        # path = '/home/wdy/DYN/notebooks/imagenet_cross_labels.txt'
        # os.makedirs(os.path.dirname(path), exist_ok=True)
        # with open(path, 'a') as f:
        #     f.write(str(len(unique_labels))+'\n')
        # 将标签转换为one-hot编码
        label_onehot = nn.functional.one_hot(inverse_indices, num_classes=len(unique_labels)).float().to(x.device)
        # 聚类均值和方差
        label_sums = torch.matmul(label_onehot.transpose(0, 1), mu_flat)
        label_counts = label_onehot.sum(dim=0, keepdim=True).transpose(0, 1) + self.eps
        cluster_mus = label_sums / label_counts

        label_vars = torch.matmul(label_onehot.transpose(0, 1), sigma2_flat)
        cluster_sigma2s = label_vars / label_counts
        
        mu_diff_squared = (mu_flat.unsqueeze(1) - cluster_mus.unsqueeze(0))**2
        weighted_mu_diff_squared = torch.einsum('blc,bl->lc', mu_diff_squared, label_onehot) / label_counts
        # mu_diff_squared = (mu_flat.unsqueeze(1) - cluster_mus.unsqueeze(0)) ** 2
        # weighted_mu_diff_squared = torch.matmul(label_onehot.transpose(0, 1), mu_diff_squared) / label_counts
        cluster_sigma2s += weighted_mu_diff_squared
        
        # 计算用于标准化的均值和方差
        cluster_mus_expanded = torch.matmul(label_onehot, cluster_mus).view(b, c, 1, 1)
        cluster_sigma2s_expanded = torch.matmul(label_onehot, cluster_sigma2s).view(b, c, 1, 1)

        # 扩展 mu_b 和 sigma2_b
        mu_b_expanded = mu_b.expand(b, -1, -1, -1)
        sigma2_b_expanded = sigma2_b.expand(b, -1, -1, -1)

        # 标准化
        x = (x - (rate_ * cluster_mus_expanded + source_rate * mu_b_expanded)) * torch.rsqrt(
            (rate_ * cluster_sigma2s_expanded + source_rate * sigma2_b_expanded) + self.eps
        )

        return x
    
    def forward(self, x):
        b, c, h, w = x.size()
        
        self.source_rate.data = torch.clamp(self.source_rate.data, 0, 1)
        mu_s = self._bn.running_mean.view(1, c, 1, 1)
        sigma2_s = self._bn.running_var.view(1, c, 1, 1)

        if self.mode == 'cluster':
            
            x_n = self.Lisc(x, mu_s, sigma2_s, self.source_rate)
            # var, mean = torch.var_mean(x, dim=[0, 2, 3], keepdim=True, unbiased=True)
            # mu = self.source_rate* mu_s + (1 - self.source_rate) * mean
            # sigma2 = self.source_rate * sigma2_s + (1 - self.source_rate) * var
            # x_n = (x - mu) * torch.rsqrt(sigma2 + self.eps)

        else:
            # print(self.mode)
            var, mean = torch.var_mean(x, dim=[0, 2, 3], keepdim=True, unbiased=True)
            mu = self.source_rate* mu_s + (1 - self.source_rate) * mean
            sigma2 = self.source_rate * sigma2_s + (1 - self.source_rate) * var
            x_n = (x - mu) * torch.rsqrt(sigma2 + self.eps)
            return self._bn(x) #SBN
                    
        weight = self._bn.weight.view(c, 1, 1)
        bias = self._bn.bias.view(c, 1, 1)
        x_n = x_n * weight + bias
        return x_n
  



class ClusterAwareBatchNorm1d(nn.Module):
    def __init__(self, num_channels , eps=1e-5, momentum=0.1, affine=True):
        super(ClusterAwareBatchNorm1d, self).__init__()
        self.mode = 'cluster'
        self.num_channels = num_channels 
        self.eps = eps
        self.affine = affine
        # BatchNorm1d for normalizing over feature channels
        self._bn = nn.BatchNorm1d(num_channels , eps=eps, momentum=momentum, affine=affine)
        self.source_rate = nn.Parameter(torch.tensor(0.8))  # learnable source rate

    def Lisc(self, x, mu_b, sigma2_b, source_rate):
        b, c = x.size()  # (batch_size, num_features)

        rate_ = 1 - source_rate
     

        # Compute the mean and variance across the batch dimension
        sigma2, mu = torch.var_mean(x, dim=1, unbiased=True)
        

        # Flatten to use clustering
        mu_flat = mu.view(b,1)
        sigma2_flat = sigma2.view(b,1)

        # Apply clustering (assuming FINCH is implemented elsewhere)
        data = mu_flat
        c1, _ = FINCH(data)  # Assuming FINCH returns cluster labels
        unique_labels, inverse_indices = torch.unique(c1, return_inverse=True)

        # One-hot encoding of cluster labels
        label_onehot = F.one_hot(inverse_indices, num_classes=len(unique_labels)).float().to(x.device)

        # Cluster statistics
        label_sums = torch.matmul(label_onehot.transpose(0, 1), mu_flat)
        label_counts = label_onehot.sum(dim=0, keepdim=True).transpose(0, 1) + self.eps
        cluster_mus = label_sums / label_counts

        label_vars = torch.matmul(label_onehot.transpose(0, 1), sigma2_flat)
        cluster_sigma2s = label_vars / label_counts

        # Adjust variance using the cluster-aware statistics
        mu_diff_squared = (mu_flat.unsqueeze(1) - cluster_mus.unsqueeze(0))**2
        weighted_mu_diff_squared = torch.einsum('blc,bl->lc', mu_diff_squared, label_onehot) / label_counts
        cluster_sigma2s += weighted_mu_diff_squared

        # Expand cluster statistics to match batch dimensions
        cluster_mus_expanded = torch.matmul(label_onehot, cluster_mus).view(b, 1)
        cluster_sigma2s_expanded = torch.matmul(label_onehot, cluster_sigma2s).view(b, 1)

        # Expand mu_b and sigma2_b
        mu_b_expanded = mu_b.expand(b, -1)
        sigma2_b_expanded = sigma2_b.expand(b, -1)

        # Normalize using cluster-aware stats
        x = (x - (rate_ * cluster_mus_expanded + source_rate * mu_b_expanded)) * torch.rsqrt(
            (rate_ * cluster_sigma2s_expanded + source_rate * sigma2_b_expanded) + self.eps
        )

        return x
    
    def forward(self, x):
        b, c = x.size()  # (batch_size, num_features)

        self.source_rate.data = torch.clamp(self.source_rate.data, 0, 1)
        mu_s = self._bn.running_mean.view(1, c)
        sigma2_s = self._bn.running_var.view(1, c)

        # Apply the cluster-aware normalization
        if self.mode == 'cluster':
            # print(self.mode)
            x_n = self.Lisc(x, mu_s, sigma2_s, self.source_rate)
            print("[info]x cluster size",len(x_n))
            
        else:
        # Scale and shift with learned affine parameters (if affine=True)
            var, mean = torch.var_mean(x, dim=1,keepdim=True, unbiased=True) 
            mu = self.source_rate* mu_s + (1 - self.source_rate) * mean
            sigma2 = self.source_rate * sigma2_s + (1 - self.source_rate) * var
            x_n = (x - mu) * torch.rsqrt(sigma2 + self.eps)
            return self._bn(x) #SBN
        # print("[info]x cluster size",len(x_n))
        weight = self._bn.weight.view(1, c)
        bias = self._bn.bias.view(1, c)
        
        # Apply affine transformation
        x_n = x_n * weight + bias
        return x_n

    
