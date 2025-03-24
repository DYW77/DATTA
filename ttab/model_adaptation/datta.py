# -*- coding: utf-8 -*-forward
from collections import deque
import copy
import functools
import math
import random
import warnings
from typing import List
import numpy as np
import torch
import torch.nn as nn
import ttab.model_adaptation.utils as adaptation_utils
from ttab.api import Batch
from ttab.loads.define_model import load_pretrained_model
from ttab.model_adaptation.base_adaptation import BaseAdaptation
from ttab.model_selection.base_selection import BaseSelection
from ttab.model_selection.metrics import Metrics
from ttab.utils.auxiliary import fork_rng_with_seed
from ttab.utils.logging import Logger
import ttab.utils.checkpoint as checkpoint
from ttab.utils.timer import Timer
from sklearn.preprocessing import MinMaxScaler
import os
import csv
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
from ttab.model_adaptation.utils import (
    DABN,
)


class DATTA(BaseAdaptation):
    """
    DATTA: Domain Diversity Aware Test-Time Adaptation for Dynamic Domain Shift Stream
    """

    def __init__(self, meta_conf, model: nn.Module):
        self.device = meta_conf.device
        super(DATTA, self).__init__(meta_conf, model)
        self.device = self._meta_conf.device
        self.entropy_loss = adaptation_utils.HLoss(
            temp_factor=self._meta_conf.temperature
        )
        self.count_backward_step = 0
        self.step = 0
        self.confusion_cache = [0]
    def _prior_safety_check(self):
        assert self._meta_conf.use_learned_stats, "DATTA uses batch-free evaluation."
        assert (
            self._meta_conf.debug is not None
        ), "The state of debug should be specified"
        assert self._meta_conf.n_train_steps > 0, "adaptation steps requires >= 1."
        
    def replace_init(self, module: nn.Module, **kwargs):
        module_output = module
        if isinstance(module, (nn.BatchNorm2d)):

            module_output = DABN(
                num_features=module.num_features,
                eps=module.eps,
                momentum=module.momentum,
                affine=module.affine,
            )
            module_output._bn.load_state_dict(module.state_dict())
            module_output.source_mu = module.running_mean.to(self._meta_conf.device)
            module_output.source_sigma2 = module.running_var.to(self._meta_conf.device)
            
        for name, child in module.named_children():
            module_output.add_module(
                name, self.replace_init(child, **kwargs)
            )
        del module
        return module_output

    def set_mode_in_DABN(self, module: nn.Module, mode, **kwargs):
        if isinstance(module, DABN):
            module.set_mode(mode)
            if mode == "single" and "single" in module._get_name():
                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)

            
        for name, child in module.named_children():
            module.add_module(
                name, self.set_mode_in_DABN(child, mode, **kwargs)
            )
     
        return module


    def _initialize_model(self, model: nn.Module):
        """Configure model for adaptation."""

        # disable grad, to (re-)enable only what specified adaptation method updates
        model = self.replace_init(model)
        model.requires_grad_(False)
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if self._meta_conf.use_learned_stats:
                    module.track_running_stats = True
                    module.momentum = self._meta_conf.bn_momentum
                else:
                    # with below, this module always uses the test batch statistics (no momentum)
                    module.track_running_stats = True

                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)

            elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)

        return model.to(self._meta_conf.device)

    def _initialize_trainable_parameters(self):
        """select target params for adaptation methods."""
        self._adapt_module_names = []
        adapt_params = []
        adapt_param_names = []

        for name_module, module in self._model.named_modules(): 
            if isinstance(module, nn.BatchNorm2d):
                self._adapt_module_names.append(name_module)
                for name_param, param in module.named_parameters():
                    adapt_params.append(param)
                    adapt_param_names.append(f"{name_module}.{name_param}")
        return adapt_params, adapt_param_names

    def one_adapt_step(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        memory_sampled_feats: torch.Tensor,
        timer: Timer,
        random_seed: int = None,
        Batch: Batch = None,
        domain_indices=None,
    ):
        self.step += 1
        """adapt the model in one step."""
        x = memory_sampled_feats.clone().detach()
        if "resnet" in self._meta_conf.model_name:
            x = model.conv1(x)
            try:
                judge, similarity = self.sum_angle(
                    x, model.bn1._bn.running_mean, model.bn1._bn.running_var
                )
            except AttributeError as e:
                print("[warning] AttributeError:", e)
                judge, similarity = self.sum_angle(
                    x, model.bn1.running_mean, model.bn1.running_var
                )

        else :
            ex_layer = 6
            for i in range(ex_layer):
                x = model.patch_embed[i](x)
            x = model.patch_embed[ex_layer].c(x)

            try:
                judge, similarity = self.sum_angle(
                    x, model.patch_embed[ex_layer].bn.source_mu, model.patch_embed[ex_layer].bn.source_sigma2
                )
            except AttributeError as e:
                print("[warning] AttributeError:", e)
                # judge, similarity = self.sum_angle(
                #     x, model.patch_embed[0].bn._bn.running_mean, model.patch_embed[0].bn._bn.running_var
                # )
        
        if judge == "single":
            model = self.set_mode_in_DABN(model,judge)
            for module in model.modules():
                if isinstance(module, (nn.BatchNorm1d)):
                    module.weight.requires_grad_(True)
                    module.bias.requires_grad_(True)

        else:
            model = self.set_mode_in_DABN(model,judge)
            model.requires_grad_(False)
            model.eval()
        
        
        with timer("forward"):
                    
            with fork_rng_with_seed(random_seed):
                y_hat = model(memory_sampled_feats)
            loss = adaptation_utils.softmax_entropy(y_hat).mean(0)
            if self.fishers is not None:
                ewc_loss = 0
                for name, param in model.named_parameters():
                    if name in self.fishers:
                        ewc_loss += (
                            self._meta_conf.fisher_alpha
                            * (
                                self.fishers[name][0]
                                * (param - self.fishers[name][1]) ** 2
                            ).sum()
                        )
                loss += ewc_loss

        with timer("backward"):
            if judge=='single':
                loss.backward()
                grads = dict(
                    (name, param.grad.clone().detach())
                    for name, param in model.named_parameters()
                    if param.grad is not None
                )
                optimizer.step()
                optimizer.zero_grad()
            else:
                loss = torch.tensor(0)
                optimizer.zero_grad()
        accuracy = (y_hat.argmax(dim=1) == Batch._y).float().mean().item()
        stats = {
            "epoch": self.step,
            "y": Batch._y.cpu().detach().numpy(),
            "accuracy": accuracy,
            "loss": loss.item(),
        }

        return {
            "optimizer": copy.deepcopy(optimizer.state_dict()),
            "loss": loss.item(),
            "yhat": y_hat,
        }

    def run_multiple_steps(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        memory_sampled_feats: torch.Tensor,
        current_batch: Batch,
        model_selection_method: BaseSelection,
        nbsteps: int,
        timer: Timer,
        random_seed: int = None,
        domain_indices=None,
    ):
        for step in range(1, nbsteps + 1):
            adaptation_result = self.one_adapt_step(
                model,
                optimizer,
                memory_sampled_feats,
                timer,
                random_seed=random_seed,
                domain_indices=domain_indices,
                Batch=current_batch,
            )
            model_selection_method.save_state(
                {
                    "model": copy.deepcopy(model.state_dict()),
                    "step": step,
                    "lr": self._meta_conf.lr,
                    **adaptation_result,
                },
                current_batch=current_batch,
            )

    def adapt_and_eval(
        self,
        episodic: bool,
        metrics: Metrics,
        model_selection_method: BaseSelection,
        current_batch: Batch,
        previous_batches: List[Batch],
        logger: Logger,
        timer: Timer,
        count: int = 0,
        domain_indices=None,
    ):
        """The key entry of test-time adaptation."""
        log = functools.partial(logger.log, display=self._meta_conf.debug)
        if episodic:
            log("\treset model to initial state during the test time.")
            self.reset()

        log(f"\tinitialize selection method={model_selection_method.name}.")
        model_selection_method.initialize()

        with timer("test_time_adaptation"):

            nbsteps = self._get_adaptation_steps(index=len(previous_batches))

            log(f"\tadapt the model for {nbsteps} steps with lr={self._meta_conf.lr}.")
            self.run_multiple_steps(
                model=self._model,
                optimizer=self._optimizer,
                memory_sampled_feats=current_batch._x,
                current_batch=current_batch,
                model_selection_method=model_selection_method,
                nbsteps=nbsteps,
                timer=timer,
                random_seed=self._meta_conf.seed,
                domain_indices=domain_indices,
            )
            # select the optimal checkpoint, and return the corresponding prediction.
            with timer("evaluate_adaptation_result"):
                optimal_state = model_selection_method.select_state()
                metrics.eval(current_batch._y, optimal_state['yhat'])
                if self._meta_conf.base_data_name in ["waterbirds"]:
                    self.tta_loss_computer.loss(
                        optimal_state['yhat'],
                        current_batch._y,
                        current_batch._g,
                        is_training=False,
                    )

                # stochastic restore part of model parameters if enabled.
                if self._meta_conf.stochastic_restore_model:
                    self.stochastic_restore()
            with timer("select_optimal_checkpoint"):
                
                log(
                    f"\tselect the optimal model ({optimal_state['step']}-th step and lr={optimal_state['lr']}) for the current mini-batch.",
                )

                self._model.load_state_dict(optimal_state["model"])
                model_selection_method.clean_up()

                if self._oracle_model_selection:
                    yhat = optimal_state["yhat"]
                    # oracle model selection needs to save steps
                    self.oracle_adaptation_steps.append(optimal_state["step"])
                    # update optimizer.
                    self._optimizer.load_state_dict(optimal_state["optimizer"])

            # print(f"\nThe Times of optimizer.step(): {self.count_backward_step}\n")

    @property
    def name(self):
        return "DATTA"


    def sum_angle(self, x, running_mean, running_var):
        b, c, h, w = x.size()
        # print(x.size())
        source_mu = running_mean.reshape(1, c, 1, 1)
        source_sigma2 = running_var.reshape(1, c, 1, 1)

        sigma2_b, mu_b = torch.var_mean(x, dim=[0, 2, 3], keepdim=True)

        mu_b = mu_b.repeat(b, 1, 1, 1)
        sigma2_b = sigma2_b.repeat(b, 1, 1, 1)
        source_mu = source_mu.repeat(b, 1, 1, 1)
        source_sigma2 = source_sigma2.repeat(b, 1, 1, 1)

        dsi = source_mu - x
        dti = mu_b - x
        dst = source_mu - mu_b

        cos_similarity = nn.CosineSimilarity(dim=1, eps=1e-10)

        similarity = cos_similarity(dsi.view(b, c, -1), dti.view(b, c, -1)).mean(1)
        similarity = similarity.cpu().detach().numpy()
        curve = np.arccos(similarity)
        curve_std = np.std(curve)
        self.confusion_cache.append(curve_std)
        
        if len(self.confusion_cache) > 50:
            if self.confusion_cache[0] == 0:
                self.confusion_cache.pop(0)
            
            peak1, peak2, valley, peak1_density, peak2_density = self.cul_peak_and_valley()
            cur_domain = self.judge_area(curve_std)

            # 定义峰值密度差异的阈值
            density_threshold_ratio = 2  # 可以根据需要调整阈值
            
            if peak2_density == 0:
                density_ratio = float('inf')
            else:
                density_ratio = peak1_density / peak2_density
                

            if density_ratio > density_threshold_ratio  or density_ratio < (1/density_threshold_ratio):
                return 'multiple', curve_std 
            else:
                if self._meta_conf.select_area == "1&4":
                    if cur_domain == "1" or cur_domain == "4":
                        return 'single', curve_std
                    else:
                        return 'multiple', curve_std
                elif self._meta_conf.select_area == "1&2":
                    if cur_domain == "1" or cur_domain == "2":
                        return 'single', curve_std
                    else:
                        return 'multiple', curve_std
                elif self._meta_conf.select_area == "2&3":
                    if cur_domain == "2" or cur_domain == "3":
                        return 'single', curve_std
                    else:
                        return 'multiple', curve_std
                elif self._meta_conf.select_area == "1":
                    if cur_domain == "1" :
                        return 'single', curve_std
                    else:
                        return 'multiple', curve_std
                elif self._meta_conf.select_area == cur_domain:
                    return 'single', curve_std
                else:
                    return 'multiple', curve_std
        return "multiple", curve_std
    
    def judge_area(self, x):
        
        peak1, peak2, valley, peak1_density, peak2_density = self.cul_peak_and_valley()
        if(x < peak1):
            return "1"
        elif (x < valley):
            return "2"
        elif (x < peak2):
            return "3"
        else:
            return "4"
    def cul_peak_and_valley(self):
        peak1 = 0
        peak2 = 0
        valley = 0
        data = np.array(self.confusion_cache)
        xmin, xmax = data.min(), data.max()
        xs = np.linspace(xmin, xmax, 1000)
        kde = gaussian_kde(data,"scott")
        kde_values = kde(xs)

        peak_indices, _ = find_peaks(kde_values)
        negative_kde_values = -kde_values
        valley_indices, _ = find_peaks(negative_kde_values)

        peak_xs = xs[peak_indices]
        valley_xs = xs[valley_indices]
        peak_densities = kde_values[peak_indices]
        valley_densities = kde_values[valley_indices]
        peak1_density, peak2_density = 0, 0

        if len(peak_densities) >= 2:
            top_two_peak_indices = np.argsort(peak_densities)[-2:]  # 返回两个最高峰的索引
            peak1_x, peak2_x = peak_xs[top_two_peak_indices]
            peak1_density, peak2_density = peak_densities[top_two_peak_indices]
            
            if peak1_x > peak2_x:
                peak1_x, peak2_x = peak2_x, peak1_x
                peak1_density, peak2_density = peak2_density, peak1_density

            peak1 = peak1_x
            peak2 = peak2_x
            # 找到这两个峰之间的最深谷值
            valley_between_peaks = []
            for i, valley_x in enumerate(valley_xs):
                if peak1_x < valley_x < peak2_x:
                    valley_between_peaks.append((valley_x, valley_densities[i]))
            
            if valley_between_peaks:
                valley, _ = min(valley_between_peaks, key=lambda x: x[1])

        return peak1, peak2, valley, peak1_density, peak2_density
