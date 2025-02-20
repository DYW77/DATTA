# -*- coding: utf-8 -*-
import copy
import os
import time
from typing import Dict, List, Type, Union
import numpy as np
import GPUtil
import psutil
from threading import Thread
import torch
import ttab.scenarios as scenarios
import ttab.utils.auxiliary as auxiliary
import ttab.utils.checkpoint as checkpoint
from ttab.api import Batch, PyTorchDataset
from ttab.loads.datasets.datasets import WBirdsDataset, group_attributes
from ttab.model_adaptation.base_adaptation import BaseAdaptation
from ttab.model_selection.base_selection import BaseSelection
from ttab.model_selection.metrics import Metrics
from ttab.utils.logging import CSVBatchLogger, Logger
from ttab.utils.timer import Timer

D = Union[torch.utils.data.Dataset, PyTorchDataset]


class Benchmark(object):
    def __init__(
        self,
        scenario: scenarios.Scenario,
        model_adaptation_cls: Type[BaseAdaptation],
        model_selection_cls: Type[BaseSelection],
        test_loader: D,
        meta_conf: Dict,
    ) -> None:
        # assign variables.
        self._scenario = scenario
        self._model_adaptation_cls = model_adaptation_cls
        self._model_selection_cls = model_selection_cls
        self._test_loader = test_loader
        self._meta_conf = copy.deepcopy(meta_conf)

        # init.
        self._safety_check()
        self._init_benchmark()
        self._memory_monitor = MemoryMonitor()
        self._memory_monitor.start()

    def _safety_check(self) -> None:
        assert hasattr(self._meta_conf, "seed")
        assert hasattr(self._meta_conf, "root_path")

        # assign variables for convenience.
        self._meta_conf.device = (
            "cuda" if not hasattr(self._meta_conf, "device") else self._meta_conf.device
        )

    def _init_benchmark(self) -> None:
        # init logging.
        self._checkpoint_path: str = checkpoint.init_checkpoint(self._meta_conf)
        self._logger = Logger(folder_path=self._checkpoint_path)
        # modify this piece of code if adding more datasets that need to compute group-wise metrics.
        if self._meta_conf.base_data_name in ["waterbirds"]:
            self._group_logger = CSVBatchLogger(
                os.path.join(self._checkpoint_path, "tta.csv"),
                n_groups=group_attributes[self._meta_conf.base_data_name],
            )
            self.tta_loss_computer = (
                self._model_adaptation_cls.construct_group_computer(
                    dataset=WBirdsDataset(
                        root=os.path.join(
                            self._meta_conf.data_path, self._meta_conf.data_names
                        ),
                        split="test",
                        device=self._meta_conf.device,
                        data_augment=False,
                    )
                )
            )
        else:
            self._group_logger = None

        # init metrics.
        self._metrics = Metrics(self._scenario)
        if self._meta_conf.record_preadapted_perf:
            self._metrics.init_auxiliary_metric(metric_name="preadapted_accuracy_top1")

        # init timer.
        self._timer = Timer(
            device=self._meta_conf.device,
            verbosity_level=1
            if not hasattr(self._meta_conf, "track_time")
            else self._meta_conf.track_time,
            log_fn=self._logger.log_metric,
            on_cuda=True if "cuda" in self._meta_conf.device else False,
        )

    @property
    def _batch_size(self):
        """This function is responsible for applying the batch-wise or sample-wise setting."""
        # it may have the internal constraints for some algos.
        assert self._scenario.test_case.data_wise in ["sample_wise", "batch_wise"]
        if self._scenario.test_case.data_wise == "sample_wise":
            return 1
        elif (
            self._scenario.test_case.data_wise
            == "batch_wise"
        ):
            return self._scenario.test_case.batch_size
        else:
            raise ValueError("invalid argument in _batch_size")

    def _offline_adapt(self):
        if self._scenario.test_case.offline_pre_adapt:
            auxiliary_loader = self._model_adaptation_cls.get_auxiliary_loader(
                scenario=self._scenario
            )
            assert (
                auxiliary_loader is not None
            ), "offline_adapt needs auxiliary_loader is not None."

            self._model_adaptation_cls.offline_adapt(
                model_selection_method=self._model_selection_cls,
                auxiliary_loader=auxiliary_loader,
                timer=self._timer,
                logger=self._logger,
                random_seed=self._meta_conf.seed,
            )

    def _online_adapt_step(
        self,
        step: int,
        epoch: int,
        batch: Batch,
        previous_batches: List[Batch],
        display_info: bool = True,
    ):
        # 记录并打印显卡信息
        def print_gpu_info():
            if torch.cuda.is_available():
                # 获取当前 GPU 的 ID (通常是 0)
                gpu_id = torch.cuda.current_device()
                # 打印显存使用情况
                print(f"GPU {gpu_id} - Allocated: {torch.cuda.memory_allocated(gpu_id) / (1024 ** 2):.2f} MB")
                print(f"GPU {gpu_id} - Reserved: {torch.cuda.memory_reserved(gpu_id) / (1024 ** 2):.2f} MB")
                print(f"GPU {gpu_id} - Max Reserved: {torch.cuda.max_memory_reserved(gpu_id) / (1024 ** 2):.2f} MB")

                # 使用 GPUtil 获取更详细的信息，包括总容量和剩余容量
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    print(f"GPU {gpu.id}: {gpu.name}")
                    print(f"  Total Memory: {gpu.memoryTotal} MB")
                    print(f"  Free Memory: {gpu.memoryFree} MB")
                    print(f"  Used Memory: {gpu.memoryUsed} MB")
            else:
                print("CUDA is not available. No GPU found.")

        with self._timer("adapt_and_eval", step=step, epoch=epoch):

            self._model_adaptation_cls.adapt_and_eval(
                episodic=self._scenario.test_case.episodic,
                metrics=self._metrics,
                model_selection_method=self._model_selection_cls,
                current_batch=batch,
                previous_batches=previous_batches,
                logger=self._logger,
                timer=self._timer,
            )
        # 在适应和评估后打印 GPU 使用情况
        # print_gpu_info()

        with self._timer("logging", step=step, epoch=epoch):
            self._logger.log_metric(
                name="runtime",
                values={
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "step": step,
                    "epoch": epoch,
                    **self._metrics.tracker.get_current_val(),
                },
                tags={"split": "test", "type": "step"},
                display=True and display_info,
            )

        # 在日志记录后再次打印 GPU 使用情况
        # print_gpu_info()

        # with self._timer("data_swap", step=step, epoch=epoch):
        #     previous_batches.append(batch.to(device="cpu"))
        
        return previous_batches

    def eval(self) -> Dict:
        self._logger.log(
            f"Test-time adaptation benchmark: scenarios={self._scenario}", display=False
        )
        self._logger.pretty_print(self._scenario)

        # safety check
        assert self._test_loader is not None

        # log dataset statistics
        if "shiftedlabel" in self._meta_conf.data_names:
            self._logger.log_metric(
                name="runtime",
                values={
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "dataset_statistics": self._test_loader.dataset.query_dataset_attr("label_statistics"),
                },
                tags={"split": "test", "type": "overall"},
                display=True,
            )

        with auxiliary.evaluation_monitor(self._meta_conf):
            # Test-time evaluation begins.
            self._offline_adapt()

            # online adaptation.
            previous_batches: List[Batch] = []
            if self._meta_conf.fishers:
                self._model_adaptation_cls.compute_fishers(
                    scenario=self._scenario, data_size=self._meta_conf.fisher_size
                )
            for step, epoch, batch in self._test_loader.iterator(
                batch_size=self._batch_size,  # apply the batch-wise or sample-wise setting.
                shuffle=False,  # we will apply shuffle operation in preparing dataset.
                repeat=False,
                ref_num_data=None,
                num_workers=self._meta_conf.num_workers
                if hasattr(self._meta_conf, "num_workers")
                else 2,
                pin_memory=True,
                drop_last=False,
                data_name=self._meta_conf.base_data_name
            ):
                # if step < 2:
                    previous_batches = self._online_adapt_step(
                        step=step,
                        epoch=epoch,
                        batch=batch,
                        previous_batches=previous_batches,
                    )

                    # if step == 100:
                    #     break
        # wtf_stats = np.array(self._model_adaptation_cls.stats_list)
        # print(f'Saving /home/wdy/DYN/logs/stats.npy')
        # np.save(f'/home/wdy/DYN/logs/stats.npy', wtf_stats)
        stats = self._metrics.tracker()
        # group-wise stats
        if self._meta_conf.base_data_name in ["waterbirds"]:
            group_stats_dict = self.tta_loss_computer.get_stats()
            tta_metrics = self.tta_loss_computer.get_target_metrics(
                group_stats_dict, self._meta_conf.group_counts
            )
            self._group_logger.log(epoch=0, batch=step, stats_dict=group_stats_dict)
            self._group_logger.flush()
            self._group_logger.close()
            stats["group_avg_accuracy"] = tta_metrics["avg_acc"] * 100
            stats["worst_group_accuracy"] = tta_metrics["robust_acc"] * 100

        self._logger.log(f"stats of test-time adaptation={stats}.")
        self._logger.log_metric(
            name="runtime",
            values={"time": time.strftime("%Y-%m-%d %H:%M:%S"), **stats},
            tags={"split": "test", "type": "overall"},
            display=True,
        )
        # 停止监控
        self._memory_monitor.stop()

        # 获取最大内存使用
        max_memory_usage = self._memory_monitor.get_max_memory_usage()
        log_cpu_memory_usage(self._logger, max_memory_usage)        # 获取最大分配的内存和最大缓存的内存
        
        max_memory_allocated = torch.cuda.max_memory_allocated(self._meta_conf.device)
        max_memory_cached = torch.cuda.max_memory_reserved(self._meta_conf.device)

        self._logger.log(f"The maximum allocated memory: {max_memory_allocated / 1024**3:.2f} GB")
        self._logger.log(f"The maximum cached memory: {max_memory_cached / 1024**3:.2f} GB")
        self._logger.log(self._timer.summary())
        
        torch.cuda.empty_cache()
        # 重置最大内存分配和缓存的统计数据
        torch.cuda.reset_max_memory_allocated(self._meta_conf.device)
        torch.cuda.reset_max_memory_cached(self._meta_conf.device)
        torch.cuda.reset_peak_memory_stats(self._meta_conf.device)
        # torch.cuda.caching_allocator_delete(self._meta_conf.device)

        max_memory_allocated = torch.cuda.max_memory_allocated(self._meta_conf.device)
        max_memory_cached = torch.cuda.max_memory_reserved(self._meta_conf.device)
        self._logger.log(f"The maximum allocated memory: {max_memory_allocated / 1024**3:.2f} GB")
        self._logger.log(f"The maximum cached memory: {max_memory_cached / 1024**3:.2f} GB")
        # torch.cuda.reset_max_memory_reserved(self._meta_conf.device)

        self._logger.save_json()
        return stats

class MemoryMonitor:
    def __init__(self):
        self.max_memory_usage = 0
        self.running = False

    def start(self):
        self.running = True
        self.thread = Thread(target=self._monitor)
        self.thread.start()

    def _monitor(self):
        process = psutil.Process()
        while self.running:
            memory_info = process.memory_info().rss
            self.max_memory_usage = max(self.max_memory_usage, memory_info)
            time.sleep(0.1)  # 每0.1秒检查一次

    def stop(self):
        self.running = False
        self.thread.join()

    def get_max_memory_usage(self):
        return self.max_memory_usage / 1024**3  # 转换为GB
    
def log_cpu_memory_usage(logger, max_memory_usage):
    logger.log(f"Maximum CPU memory usage during algorithm: {max_memory_usage:.2f} GB")