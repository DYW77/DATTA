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
# from ttab.model_adaptation.Our import replace_bn
from ttab.model_adaptation.utils import (
    # CustomBatchNorm,
    DynamicBatchNorm
    # HomoDynamicBatchNorm,
    # HeterDynamicBatchNorm,
)

# from ttab.utils.tsne import (
#     save_tsne_results,
#     register_hook_on_fc_layer,
#     tsne_visualize,
#     extracted_features, extracted_labels
# )


class DATTA(BaseAdaptation):
    """
    DATTA: Robust Continual Test-time Adaptation Against Temporal Correlation,
    https://arxiv.org/abs/2208.05117,
    https://github.com/TaesikGong/DATTA
    """

    def __init__(self, meta_conf, model: nn.Module):
        self.device = meta_conf.device
        super(DATTA, self).__init__(meta_conf, model)
        self.device = self._meta_conf.device
        # self.memory = self.define_memory()
        self.entropy_loss = adaptation_utils.HLoss(
            temp_factor=self._meta_conf.temperature
        )
        self.count_backward_step = 0
        self.step = 0
        # self.confusion_cache = deque([0.0], maxlen=1000) 
        self.confusion_cache = [0]
        # print(meta_conf)
        # path = (
        #     self._meta_conf.root_path
        #     + f"/sorted_test_{self._meta_conf.base_data_name}_{self._meta_conf.inter_domain}_cosine_similartiy_test_batch.csv"
        # )
        # self.confusion_indicator=0.0
    def _prior_safety_check(self):
        assert self._meta_conf.use_learned_stats, "DATTA uses batch-free evaluation."
        assert (
            self._meta_conf.debug is not None
        ), "The state of debug should be specified"
        assert self._meta_conf.n_train_steps > 0, "adaptation steps requires >= 1."
        
    def replace_init(self, module: nn.Module, **kwargs):
        module_output = module
        if isinstance(module, (nn.BatchNorm2d)):

            module_output = DynamicBatchNorm(
                num_features=module.num_features,
                eps=module.eps,
                momentum=module.momentum,
                affine=module.affine,
            )
            # module_output._bn = copy.deepcopy(module)
            module_output._bn.load_state_dict(module.state_dict())
            # module_output.heter_bn.load_state_dict(module.state_dict())
            module_output.source_mu = module.running_mean.to(self._meta_conf.device)
            module_output.source_sigma2 = module.running_var.to(self._meta_conf.device)
            
        for name, child in module.named_children():
            # print(name, child._get_name())
            module_output.add_module(
                name, self.replace_init(child, **kwargs)
            )
        del module
        return module_output

    def set_mode_in_DABN(self, module: nn.Module, mode, **kwargs):
        if isinstance(module, DynamicBatchNorm):
            module.set_mode(mode)
            if mode == "homo" and "homo" in module._get_name():
                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)

            
        for name, child in module.named_children():
            # print(name, child._get_name())
            module.add_module(
                name, self.set_mode_in_DABN(child, mode, **kwargs)
            )
     
        return module

    
    # def replace_DynamicBatchNorm(self, module: nn.Module, **kwargs):
    #     """
    #     Recursively convert all BatchNorm to HeterDynamicBatchNorm.
    #     """
    #     module_output = module
        
    #     if isinstance(module, HomoDynamicBatchNorm):
    #         module_output = HeterDynamicBatchNorm(
    #             num_features=module._bn.num_features,
    #             eps=module.eps,
    #             momentum=module.momentum,
    #             affine=module._bn.affine,
    #         )
    #         # module_output._bn = copy.deepcopy(module._bn)
    #         module_output.source_mu = module.source_mu
    #         module_output.source_sigma2 = module.source_sigma2
    #         module_output.domain_difference_sum = module.domain_difference_sum
    #         module_output._bn.load_state_dict(module._bn.state_dict())
    #     elif isinstance(module, HeterDynamicBatchNorm):
    #         return module

    #     for name, child in module.named_children():
    #         # print(name, child._get_name())
    #         module_output.add_module(
    #             name, self.replace_HeterDynamicBatchNorm(child, **kwargs)
    #         )
    #     del module
    #     return module_output

    def _initialize_model(self, model: nn.Module):
        """Configure model for adaptation."""

        # disable grad, to (re-)enable only what specified adaptation method updates
        model = self.replace_init(model)
        # register_hook_on_fc_layer(model, fc_layer_name="classifier")

        model.requires_grad_(False)
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if self._meta_conf.use_learned_stats:
                    module.track_running_stats = True
                    module.momentum = self._meta_conf.bn_momentum
                else:
                    # with below, this module always uses the test batch statistics (no momentum)
                    module.track_running_stats = True
                    # module.running_mean = None
                    # module.running_var = None

                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)

            elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)

        return model.to(self._meta_conf.device)

    # def _initialize_trainable_parameters(self):
    #     """select target params for adaptation methods."""
    #     self._adapt_module_names = []
    #     adapt_params = []
    #     adapt_param_names = []

    #     for name_module, module in self._model.named_children():
    #         self._adapt_module_names.append(name_module)
    #         for name_param, param in module.named_parameters():
    #             adapt_params.append(param)
    #             adapt_param_names.append(f"{name_module}.{name_param}")

    #     return adapt_params, adapt_param_names
    def _initialize_trainable_parameters(self):
        """select target params for adaptation methods."""
        self._adapt_module_names = []
        adapt_params = []
        adapt_param_names = []

        for name_module, module in self._model.named_modules():  # 使用 named_modules 而不是 named_children
            # 检查模块名称是否包含 "homo" 并且模块是 BatchNorm 类型
            if isinstance(module, nn.BatchNorm2d):
                self._adapt_module_names.append(name_module)
                for name_param, param in module.named_parameters():
                    adapt_params.append(param)
                    adapt_param_names.append(f"{name_module}.{name_param}")
        print("[info]adapt_param_names:",adapt_param_names)
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
        # judge="",
    ):
        # global extracted_features  # 引用全局变量
        # global extracted_labels
        self.step += 1
        """adapt the model in one step."""
        # just for DATTA
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
            # print(model.patch_embed)
            ex_layer = 6
            for i in range(ex_layer):
                x = model.patch_embed[i](x)
            x = model.patch_embed[ex_layer].c(x)

            try:
                # 假设'model'是你的EfficientViT模型实例，'x'是你的输入张量
                judge, similarity = self.sum_angle(
                    x, model.patch_embed[ex_layer].bn.source_mu, model.patch_embed[ex_layer].bn.source_sigma2
                )
                # judge, similarity = self.sum_angle(
                #     x, model.patch_embed[ex_layer].bn.homo_bn.running_mean, model.patch_embed[ex_layer].bn.homo_bn.running_var
                # )
            except AttributeError as e:
                print("[warning] AttributeError:", e)
                # 如果上述属性不存在，尝试直接访问bn层的属性
                # judge, similarity = self.sum_angle(
                #     x, model.patch_embed[0].bn._bn.running_mean, model.patch_embed[0].bn._bn.running_var
                # )
        
        # model = self.set_mode_in_DABN(model,"homo")
        # model = self.set_mode_in_DABN(model,"heter")
        if judge == "homo":
            model = self.set_mode_in_DABN(model,judge)
            for module in model.modules():
                if isinstance(module, (nn.BatchNorm1d)):
                    module.weight.requires_grad_(True)
                    module.bias.requires_grad_(True)

        else:
            model = self.set_mode_in_DABN(model,judge)
            model.requires_grad_(False)
            # model.requires_grad_(False)
            model.eval()
        
        # model.to(self._meta_conf.device)
        
        with timer("forward"):
                    
            with fork_rng_with_seed(random_seed):
                y_hat = model(memory_sampled_feats)
            loss = adaptation_utils.softmax_entropy(y_hat).mean(0)
            # loss = self.entropy_loss(y_hat)

            # apply fisher regularization when enabled
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
            # extracted_labels.append(Batch._y.clone().detach())  # 保存当前批次的标签

        with timer("backward"):
            if judge=='homo':
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
                # model.zero_grad()
                # grads = dict(
                #     (name, param.grad.clone().detach())
                #     for name, param in model.named_parameters()
                #     if param.grad is not None
                # )
                optimizer.zero_grad()
        accuracy = (y_hat.argmax(dim=1) == Batch._y).float().mean().item()
        # log(f"\taccuracy={accuracy:.4f}")
        stats = {
            "epoch": self.step,
            "y": Batch._y.cpu().detach().numpy(),
            "accuracy": accuracy,
            # "similarity": similarity.mean(0).tolist(),
            "loss": loss.item(),
            # "grads": grads,
            # "yhat": y_hat.argmax(dim=1).cpu().detach().numpy(),
            # "domain_indices": np.array(domain_indices),
        }
        # self.logger.log(stats)

        return {
            "optimizer": copy.deepcopy(optimizer.state_dict()),
            "loss": loss.item(),
            # "grads": grads,
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
        # x = memory_sampled_feats.clone().detach()
        # if "resnet" in self._meta_conf.model_name:
        #     x = model.conv1(x)
        #     try:
        #         judge, similarity = self.sum_angle(
        #             x, model.bn1._bn.running_mean, model.bn1._bn.running_var
        #         )
        #     except AttributeError as e:
        #         print("[warning] AttributeError:", e)
        #         judge, similarity = self.sum_angle(
        #             x, model.bn1.running_mean, model.bn1.running_var
        #         )

        # else :
        #     print(model.patch_embed)
        #     ex_layer = 6
        #     for i in range(ex_layer):
        #         x = model.patch_embed[i](x)
        #     x = model.patch_embed[ex_layer].c(x)
        #     # x = model.patch_embed[0](x)
        #     # x = model.patch_embed[1](x)
        #     # x = model.patch_embed[2].c(x)
        #     # print(model.patch_embed[0].bn._bn)
        #     try:
        #         # 假设'model'是你的EfficientViT模型实例，'x'是你的输入张量
        #         judge, similarity = self.sum_angle(
        #             x, model.patch_embed[ex_layer].bn._bn.running_mean, model.patch_embed[ex_layer].bn._bn.running_var
        #         )
        #     except AttributeError as e:
        #         print("[warning] AttributeError:", e)
        #         # 如果上述属性不存在，尝试直接访问bn层的属性
        #         # judge, similarity = self.sum_angle(
        #         #     x, model.patch_embed[0].bn._bn.running_mean, model.patch_embed[0].bn._bn.running_var
        #         # )
        

        # if judge == "homo":
        #     model = self.replace_HomoDynamicBatchNorm(model)
        #     for module in model.modules():
        #         if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
        #             module.weight.requires_grad_(True)
        #             module.bias.requires_grad_(True)

        # else:
        #     model = self.replace_HeterDynamicBatchNorm(model)
        #     model.requires_grad_(False)
        #     model.eval()
        
        # model.to(self._meta_conf.device)
        for step in range(1, nbsteps + 1):
            adaptation_result = self.one_adapt_step(
                model,
                optimizer,
                memory_sampled_feats,
                timer,
                random_seed=random_seed,
                domain_indices=domain_indices,
                Batch=current_batch,
                # judge=judge,
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
        # global extracted_features  # 引用全局变量
        # global extracted_labels
        """The key entry of test-time adaptation."""
        # some simple initialization.
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
                # memory_sampled_feats=memory_sampled_feats,
                # change
                current_batch=current_batch,
                model_selection_method=model_selection_method,
                nbsteps=nbsteps,
                timer=timer,
                random_seed=self._meta_conf.seed,
                domain_indices=domain_indices,
            )
            # chunk = 50
            # if len(extracted_features) > 0 and len(extracted_features) % chunk == 0:
            #     features = torch.cat(extracted_features[-chunk:], dim=0)
            #     labels = torch.cat(extracted_labels[-chunk:], dim=0)
            #     root1 = f"{self._meta_conf.root_path}/tsne/plg"
            #     root2 = f"{self._meta_conf.root_path}/tsne/npy"
            #     os.makedirs(root1, exist_ok=True)
            #     os.makedirs(root2, exist_ok=True)
            #     # 调用 t-SNE 可视化
            #     # tsne_visualize(
            #     #     features,
            #     #     labels,
            #     #     title=f"{self._meta_conf.model_adaptation_method}'s t-SNE Visualization in {self._meta_conf.base_data_name}",
            #     #     save_path=f"{root1}/{self.step}.png"
            #     # )
            #     # 保存 t-SNE 结果
            #     # save_tsne_results(
            #     #     features,
            #     #     labels,
            #     #     save_path=f"{root2}/{self.step}.npy"
            #     # )
            #     extracted_features.clear()
            #     extracted_labels.clear()   

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

            print(f"\nThe Times of optimizer.step(): {self.count_backward_step}\n")

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
        # print("[info]curve:",curve.mean())
        # 保存curve数据到txt文件
        # 将数值转换为字符串，并格式化为浮点数格式
        # curve_str = f"{curve.mean():.18f}"

        # # 打开文件，使用 'a' 模式以追加模式打开
        # with open('curve_data_homo.txt', 'a') as file:
        # # with open('curve_data_cross.txt', 'a') as file:
        #     # 将数值字符串写入文件，每个数值后面添加一个换行符
        #     file.write(curve_str + '\n')
        # print(f"[info] curve_std:{curve_std}")
        self.confusion_cache.append(curve_std)
        # print(f"[info] self.confusion_cache:{len(self.confusion_cache)}")
        
        if len(self.confusion_cache) > 50:
            if self.confusion_cache[0] == 0:
                self.confusion_cache.pop(0)
                # self.confusion_cache.popleft()
            
            peak1, peak2, valley, peak1_density, peak2_density = self.cul_peak_and_valley()
            cur_domain = self.judge_area(curve_std)
            # print(f"[info] peak1={peak1}, valley={valley}, peak2={peak2}, peak1_density={peak1_density}, peak2_density={ peak2_density}")
            # print(f"DS: {curve_std}, domain: {self.judge_area(curve_std)}")

            # 定义峰值密度差异的阈值
            density_threshold_ratio = 2  # 可以根据需要调整阈值
            
            # density = True
            # 判断两个峰值的密度差异是否超过阈值
            if peak2_density == 0:
                density_ratio = float('inf')
            else:
                density_ratio = peak1_density / peak2_density
                # if density_ratio > density_threshold_ratio or density_ratio < (1/density_threshold_ratio):
                #     density = False
                

            if density_ratio > density_threshold_ratio  or density_ratio < (1/density_threshold_ratio):
                # print("[info] 两个峰的密度差异过大，停止更新。")
                return 'heter', curve_std  # 可以定义一个新的返回值 'stop'
            else:
                if self._meta_conf.select_area == "1&4":
                    if cur_domain == "1" or cur_domain == "4":
                        return 'homo', curve_std
                    else:
                        return 'heter', curve_std
                elif self._meta_conf.select_area == "1&2":
                    if cur_domain == "1" or cur_domain == "2":
                        return 'homo', curve_std
                    else:
                        return 'heter', curve_std
                elif self._meta_conf.select_area == "2&3":
                    if cur_domain == "2" or cur_domain == "3":
                        return 'homo', curve_std
                    else:
                        return 'heter', curve_std
                elif self._meta_conf.select_area == "1":
                    if cur_domain == "1" :
                        return 'homo', curve_std
                    else:
                        return 'heter', curve_std
                elif self._meta_conf.select_area == cur_domain:
                    return 'homo', curve_std
                else:
                    return 'heter', curve_std
        print(np.mean(self.confusion_cache))
        return "heter", curve_std
    
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
        # 计算data的L2范数
        # data = np.linalg.norm(data, ord=2)
        # 计算KDE
        xmin, xmax = data.min(), data.max()
        xs = np.linspace(xmin, xmax, 1000)
        kde = gaussian_kde(data)
        kde_values = kde(xs)

        # 找出峰值和谷值
        peak_indices, _ = find_peaks(kde_values)
        negative_kde_values = -kde_values
        valley_indices, _ = find_peaks(negative_kde_values)

        peak_xs = xs[peak_indices]
        valley_xs = xs[valley_indices]
        peak_densities = kde_values[peak_indices]
        valley_densities = kde_values[valley_indices]
        peak1_density, peak2_density = 0, 0

        if len(peak_densities) >= 2:
            # 获取最高的两个峰值及其索引
            top_two_peak_indices = np.argsort(peak_densities)[-2:]  # 返回两个最高峰的索引
            peak1_x, peak2_x = peak_xs[top_two_peak_indices]
            peak1_density, peak2_density = peak_densities[top_two_peak_indices]
            
            # 确保 peak1_x < peak2_x
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
