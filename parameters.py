# -*- coding: utf-8 -*-
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    # define meta info (this part can be ignored if `monitor` package is unused).
    parser.add_argument("--job_name", default="tta", type=str)
    parser.add_argument("--job_id", default=None, type=str)
    parser.add_argument("--timestamp", default=None, type=str)
    parser.add_argument("--python_path", default="python", type=str)
    parser.add_argument("--main_file", default="run_exp.py", type=str)

    parser.add_argument("--script_path", default=None, type=str)
    parser.add_argument("--script_class_name", default=None, type=str)
    parser.add_argument("--num_jobs_per_node", default=2, type=int)
    parser.add_argument("--num_jobs_per_script", default=1, type=int)
    parser.add_argument("--wait_in_seconds_per_job", default=15, type=float)
    
    # define test evaluation info.
    parser.add_argument("--root_path", default="./domain_nums_exp", type=str)
    parser.add_argument("--data_path", default="/home/wdy/Exp/datasets", type=str)
    parser.add_argument(
        "--ckpt_path",
        default="/home/wdy/Exp/model/rn50_bn_cifar100.pth",
        # default="/home/wdy/Exp/model/rn26_bn_ssh_cifar100.pth",
        type=str,
    )
    parser.add_argument("--seed", default=2022, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--num_cpus", default=2, type=int)

    # define the task & model & adaptation & selection method.
    parser.add_argument("--model_name", default="resnet50", type=str)
    parser.add_argument("--group_norm_num_groups", default=None, type=int)
    parser.add_argument(
        "--model_adaptation_method",
        default="bn_adapt",
        choices=[
            "no_adaptation",
            "tent",
            "bn_adapt",
            # "memo",
            # "shot",
            # "t3a",
            # "ttt",
            "note",
            "sar",
            # "conjugate_pl",
            # "cotta",
            "rotta",
            "eata",
            "deyo",
            "cntta",
            "tent_iabn",
            "vida",
        ],
        type=str,
    )
    parser.add_argument(
        "--model_selection_method",
        default="last_iterate",
        choices=["last_iterate", "oracle_model_selection"],
        type=str,
    )
    parser.add_argument("--task", default="classification", type=str)

    # define the test scenario.
    parser.add_argument("--test_scenario", default=None, type=str)
    """
    cifar10
    """
    parser.add_argument(
        "--base_data_name",
        default="cifar10",
        choices=[
            "cifar10",
            "cifar100",
            "imagenet",
            "officehome",
            "pacs",
            "coloredmnist",
            "waterbirds",

        ],
        type=str,
    )
    parser.add_argument("--src_data_name", default="cifar10", type=str)
    parser.add_argument(
        "--data_names", default=
                                "cifar10_c_deterministic-gaussian_noise-5;"
                                "cifar10_c_deterministic-shot_noise-5;"
                                "cifar10_c_deterministic-impulse_noise-5;"
                                "cifar10_c_deterministic-defocus_blur-5;"
                                "cifar10_c_deterministic-glass_blur-5;"
                                "cifar10_c_deterministic-motion_blur-5;"
                                "cifar10_c_deterministic-zoom_blur-5;"
                                "cifar10_c_deterministic-snow-5;"
                                "cifar10_c_deterministic-frost-5;"
                                "cifar10_c_deterministic-fog-5;"
                                "cifar10_c_deterministic-brightness-5;"
                                "cifar10_c_deterministic-contrast-5;"
                                "cifar10_c_deterministic-elastic_transform-5;"
                                "cifar10_c_deterministic-pixelate-5;"
                                "cifar10_c_deterministic-jpeg_compression-5"


                                # "cifar10_c_deterministic-pixelate-5"
                                # "cifar10_c_deterministic-gaussian_noise-5"
        , type=str
    )
    """
    cifar100
    """
    # parser.add_argument(
    #     "--base_data_name",
    #     default="cifar100",
    #     choices=[
    #         "cifar10",
    #         "cifar100",
    #         "imagenet",
    #         "officehome",
    #         "pacs",
    #         "coloredmnist",
    #         "waterbirds",

    #     ],
    #     type=str,
    # )
    # parser.add_argument("--src_data_name", default="cifar100", type=str)
    # parser.add_argument(
    #     "--data_names", default=
    #                             "cifar100_c_deterministic-gaussian_noise-5;"
    #                             "cifar100_c_deterministic-shot_noise-5;"
    #                             "cifar100_c_deterministic-impulse_noise-5;"
    #                             "cifar100_c_deterministic-defocus_blur-5;"
    #                             "cifar100_c_deterministic-glass_blur-5;"
    #                             "cifar100_c_deterministic-motion_blur-5;"
    #                             "cifar100_c_deterministic-zoom_blur-5;"
    #                             "cifar100_c_deterministic-snow-5;"
    #                             "cifar100_c_deterministic-frost-5;"
    #                             "cifar100_c_deterministic-fog-5;"
    #                             "cifar100_c_deterministic-brightness-5;"
    #                             "cifar100_c_deterministic-contrast-5;"
    #                             "cifar100_c_deterministic-elastic_transform-5;"
    #                             "cifar100_c_deterministic-pixelate-5;"
    #                             "cifar100_c_deterministic-jpeg_compression-5",
    #      type=str
    # )
    """
    imagenet
    """
    # parser.add_argument(
    #     "--base_data_name",
    #     default="imagenet",
    #     choices=[
    #         "cifar10",
    #         "cifar100",
    #         "imagenet",
    #         "officehome",
    #         "pacs",
    #         "coloredmnist",
    #         "waterbirds",

    #     ],
    #     type=str,
    # )
    # parser.add_argument("--src_data_name", default="imagenet", type=str)
    # parser.add_argument(
    #     "--data_names", default=
    #                             "imagenet_c_deterministic-gaussian_noise-5;"
    #                             "imagenet_c_deterministic-shot_noise-5;"
    #                             "imagenet_c_deterministic-impulse_noise-5;"
    #                             "imagenet_c_deterministic-defocus_blur-5;"
    #                             "imagenet_c_deterministic-glass_blur-5;"
    #                             "imagenet_c_deterministic-motion_blur-5;"
    #                             "imagenet_c_deterministic-zoom_blur-5;"
    #                             "imagenet_c_deterministic-snow-5;"
    #                             "imagenet_c_deterministic-frost-5;"
    #                             "imagenet_c_deterministic-fog-5;"
    #                             "imagenet_c_deterministic-brightness-5;"
    #                             "imagenet_c_deterministic-contrast-5;"
    #                             "imagenet_c_deterministic-elastic_transform-5;"
    #                             "imagenet_c_deterministic-pixelate-5;"
    #                             "imagenet_c_deterministic-jpeg_compression-5"
    #     , type=str
    # )
    parser.add_argument(
        "--data_wise",
        default="batch_wise",
        choices=["batch_wise", "sample_wise"],
        type=str,
    )
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--n_train_steps", default=1, type=int)
    parser.add_argument("--offline_pre_adapt", default=False, type=str2bool)
    parser.add_argument("--episodic", default=False, type=str2bool)
    parser.add_argument("--intra_domain_shuffle", default=True, type=str2bool)
    parser.add_argument(
        "--inter_domain",
        default="BatchMixing",
        choices=[
            "HomogeneousNoMixture",
            "HeterogeneousNoMixture",
            "InOutMixture",
            "CrossMixture",
            "CrossHeterMixture",
            "BatchMixing",
        ],
        type=str,
    )
    ##################################
    parser.add_argument("--sp_order", 
        default="Shuffle_Shuffle",
        help="Interval/Shuffle: scenarios appear on turn or not,"
             "Seq/Shuffle: For Homo scenario, shuffle Homo or not",
        choices=[
            "Interval_Seq",
            "Interval_Shuffle",
            "Shuffle_Shuffle",
        ],
        type=str,
    )
    parser.add_argument("--sp_scenarios", 
        default="Homo&Cross&Heter",
        help="How to mix batch",
        choices=[
            "Homo&Cross",
            "Homo&Heter",
            "Cross&Heter",
            "Homo&Cross&Heter",
        ],
        type=str,
    )
    ##################################
    # Test domain
    parser.add_argument("--domain_sampling_name", default="uniform", type=str)
    parser.add_argument("--domain_sampling_ratio", default=1, type=float)
    # HeterogeneousNoMixture
    parser.add_argument("--non_iid_pattern", default="class_wise_over_domain", type=str)
    parser.add_argument("--non_iid_ness", default=0.01, type=float)
    # for evaluation.
    # label shift
    parser.add_argument(
        "--label_shift_param",
        help="parameter to control the severity of label shift",
        default=None,
        type=float,
    )
    parser.add_argument(
        "--data_size",
        help="parameter to control the size of dataset",
        default=None,
        type=int,
    )
    # optimal model selection
    parser.add_argument(
        "--step_ratios",
        nargs="+",
        default=[0.1, 0.3, 0.5, 0.75],
        help="ratios used to control adaptation step length",
        type=float,
    )
    parser.add_argument("--step_ratio", default=None, type=float)
    # time-varying
    parser.add_argument("--stochastic_restore_model", default=False, type=str2bool)
    parser.add_argument("--restore_prob", default=0.01, type=float)
    parser.add_argument("--fishers", default=False, type=str2bool)
    parser.add_argument(
        "--fisher_size",
        default=5000,
        type=int,
        help="number of samples to compute fisher information matrix.",
    )
    parser.add_argument(
        "--fisher_alpha",
        type=float,
        default=1.5,
        help="the trade-off between entropy and regularization loss",
    )
    # method-wise hparams
    parser.add_argument(
        "--aug_size",
        default=32,
        help="number of per-image augmentation operations in memo and ttt",
        type=int,
    )
    parser.add_argument(
        "--entry_of_shared_layers",
        default=None,
        help="the split position of auxiliary head. Only used in TTT.",
    )
    # metrics
    parser.add_argument(
        "--record_preadapted_perf",
        default=False,
        help="record performance on the local batch prior to implementing test-time adaptation.",
        type=str2bool,
    )
    # misc
    parser.add_argument(
        "--grad_checkpoint",
        default=False,
        help="Trade computation for gpu space.",
        type=str2bool,
    )
    parser.add_argument("--debug", default=False, help="Display logs.", type=str2bool)

    # parse conf.
    conf = parser.parse_args()
    return conf


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ValueError("Boolean value expected.")


if __name__ == "__main__":
    args = get_args()
