import parameters
import run_exp as run
from tqdm import tqdm
import gc
import threading
import ttab.configs.algorithms as agrth
from PIL import ImageFile
# nums=[3,6,9,12,15]
# bs1 = [16]#,2,4,8,16
# bs2 = [2,4,8]
data_names_choice = [
    # [   
    #     "cifar10_c_deterministic-gaussian_noise-5;"
    #     "cifar10_c_deterministic-shot_noise-5;"
    #     "cifar10_c_deterministic-impulse_noise-5;"
    #     "cifar10_c_deterministic-defocus_blur-5;"
    #     "cifar10_c_deterministic-glass_blur-5;"
    #     "cifar10_c_deterministic-motion_blur-5;"
    #     "cifar10_c_deterministic-zoom_blur-5;"
    #     "cifar10_c_deterministic-snow-5;"
    #     "cifar10_c_deterministic-frost-5;"
    #     "cifar10_c_deterministic-fog-5;"
    #     "cifar10_c_deterministic-brightness-5;"
    #     "cifar10_c_deterministic-contrast-5;"
    #     "cifar10_c_deterministic-elastic_transform-5;"
    #     "cifar10_c_deterministic-pixelate-5;"
    #     "cifar10_c_deterministic-jpeg_compression-5"
    # ,],
    # [
    #     "cifar100_c_deterministic-gaussian_noise-5;"
    #     "cifar100_c_deterministic-shot_noise-5;"
    #     "cifar100_c_deterministic-impulse_noise-5;"
    #     "cifar100_c_deterministic-defocus_blur-5;"
    #     "cifar100_c_deterministic-glass_blur-5;"
    #     "cifar100_c_deterministic-motion_blur-5;"
    #     "cifar100_c_deterministic-zoom_blur-5;"
    #     "cifar100_c_deterministic-snow-5;"
    #     "cifar100_c_deterministic-frost-5;"
    #     "cifar100_c_deterministic-fog-5;"
    #     "cifar100_c_deterministic-brightness-5;"
    #     "cifar100_c_deterministic-contrast-5;"
    #     "cifar100_c_deterministic-elastic_transform-5;"
    #     "cifar100_c_deterministic-pixelate-5;"
    #     "cifar100_c_deterministic-jpeg_compression-5"
    # ,],
    [        
        "imagenet_c_deterministic-gaussian_noise-5;"
             "imagenet_c_deterministic-shot_noise-5;"
             "imagenet_c_deterministic-impulse_noise-5;"
             "imagenet_c_deterministic-defocus_blur-5;"
             "imagenet_c_deterministic-glass_blur-5;"
             "imagenet_c_deterministic-motion_blur-5;"
             "imagenet_c_deterministic-zoom_blur-5;"
             "imagenet_c_deterministic-snow-5;"
             "imagenet_c_deterministic-frost-5;"
             "imagenet_c_deterministic-fog-5;"
             "imagenet_c_deterministic-brightness-5;"
             "imagenet_c_deterministic-contrast-5;"
             "imagenet_c_deterministic-elastic_transform-5;"
             "imagenet_c_deterministic-pixelate-5;"
             "imagenet_c_deterministic-jpeg_compression-5"
             ,]
]
all_data_name = [
    #  "cifar10",
    # "cifar100",    
    "imagenet"
    ]
#,"imagenet"  # ,"imagenet","cifar100","cifar100"
seed = [
        2022,
        # 2023,
        # 2024
        ]
# ckpt_path = [
# "/home/young/ttab-benchmarkData/rn50_bn_cifar10.pth",
# "/home/young/ttab-benchmarkData/rn50_bn_cifar100.pth",
# ]
corruption_domain = [
    # "HomogeneousNoMixture",
    # "CrossMixture",
    "BatchMixing",

]
sp_order=[
    "Interval_Seq", 
    # "Shuffle_Shuffle"
]
adaptation_method_0 = [
            "datta",
            # "sar",
            # "deyo",
            # "rotta",
            # "no_adaptation",
            # "tent",
            # "note",
            # "vida",
]

def one_test_thread():
    for sp in sp_order:
        for method in adaptation_method_0:
            for s in seed:
                for domain_type in corruption_domain:
                    for i, domain_order in enumerate(data_names_choice):
                        # for batch_size in bs1:
                            args_note = parameters.get_args()
                            args_note.model_adaptation_method = method
                            args_note.device = "cuda:0"
                            args_note.batch_size = 64
                            args_note.model_name="efficientvit_224"#"resnet50"
                            # args_note.model_name="resnet50"
                            # args_note.lr=1e-4
                            args_note.non_iid_ness = 0.01
                            args_note.domain_sampling_ratio = 1
                            args_note.corruption_num = "15"
                            # args_note.select_area="1&2"
                            args_note.select_area="1&2"
                            args_note.seed = 2022
                            args_note.inter_domain = domain_type
                            args_note.data_names = domain_order[0]
                            args_note.base_data_name = all_data_name[i]
                            args_note.src_data_name = all_data_name[i]
                            args_note.sp_order = sp
                            args_note.sp_scenarios = "Homo&Cross"
                            if args_note.base_data_name  == "cifar10":
                                args_note.lr=1e-4
                                # args_note.data_names = domain_order[0]
                                args_note.ckpt_path = "/home/wdy/Exp/model/efficientvit/cifar10.pth"
                                # args_note.ckpt_path = "/home/wdy/Exp/model/rn50_bn_cifar10.pth"
                            elif args_note.base_data_name  == "cifar100":
                                # args_note.data_names = domain_order[0]
                                args_note.lr=1e-4
                                args_note.ckpt_path = "/home/wdy/Exp/model/efficientvit/cifar100.pth"
                                # args_note.ckpt_path = "/home/wdy/Exp/model/rn50_bn_cifar100.pth"
                            elif args_note.base_data_name  == "imagenet":
                                # args_note.data_names = domain_order[0]
                                args_note.domain_sampling_ratio = 0.05
                                args_note.lr=1e-5
                                args_note.ckpt_path = "/home/wdy/Exp/model/efficientvit/m5.pth"
                            # args_note.job_id = (f"{method}_{domain_type}_{args_note.data_names}")
                            args_note.job_name = f"{all_data_name[i]}-{method}-{domain_type}-{sp}"#
                            # args_note.root_path = f"./logs/ICME_daft_diffLR_{s}/{args_note.job_name}"
                            args_note.root_path = f"./logs/ICME_rebuttal_kde_{s}/{args_note.job_name}"
                            print(
                                f"Running experiment for domain: {domain_type},\nadaptation method: note,\ndata names: {domain_order},\nmodel_path:{args_note.ckpt_path}\n"
                            )
                            # print()
                            run.main(init_config=args_note)
                            print(f'mission: {args_note.job_id} done!')
        print("Exp has finished.")


# def two_test_thread():
#         for method in adaptation_method_1:
#             # for s in seed:
#                 for domain_type in corruption_domain:
#                     for i, domain_order in enumerate(data_names_choice):
#                         # for batch_size in bs1:
#                             args_note = parameters.get_args()
#                             args_note.model_adaptation_method = method
#                             args_note.device = "cuda:1"
#                             args_note.batch_size = 1
#                             args_note.model_name="resnet50"
#                             args_note.lr=1e-4
#                             args_note.non_iid_ness = 0.01
#                             args_note.domain_sampling_ratio = 1
#                             args_note.corruption_num = "15"
#                             args_note.seed = 2022
#                             args_note.inter_domain = domain_type
#                             # print("!!!!!!!!!!!!!!!!!!!!", all_data_name[i])
#                             args_note.data_names = domain_order[0]
#                             args_note.base_data_name = all_data_name[i]
#                             args_note.src_data_name = all_data_name[i]
#                             if args_note.base_data_name  == "cifar10":
#                                 # args_note.data_names = domain_order[0]
#                                 args_note.ckpt_path = "/home/wdy/Exp/model/rn50_bn_cifar10.pth"
#                             elif args_note.base_data_name  == "cifar100":
#                                 # args_note.data_names = domain_order[0]
#                                 args_note.ckpt_path = "/home/wdy/Exp/model/rn50_bn_cifar100.pth"
#                             elif args_note.base_data_name  == "imagenet":
#                                 # args_note.data_names = domain_order[0]
#                                 args_note.domain_sampling_ratio = 0.1
#                                 args_note.ckpt_path = "/home/wdy/Exp/model/rn50_bn_cifar100.pth"
#                             # args_note.job_id = (f"{method}_{domain_type}_{args_note.data_names}")
#                             args_note.job_name = f"{all_data_name[i]}-{method}"#-{domain_type}
#                             args_note.root_path = f"./logs/batchsize/{args_note.job_name}"
#                             print(
#                                 f"Running experiment for domain: {domain_type},\nadaptation method: note,\ndata names: {domain_order},\nmodel_path:{args_note.ckpt_path}\n"
#                             )
#                             # print()
#                             run.main(init_config=args_note)
#                             print(f'mission: {args_note.job_id} done!')
#         print("Exp has finished.")



sq_num = 0
t1 = threading.Thread(target=one_test_thread)
t1.start()

# t2 = threading.Thread(target=two_test_thread)
# t2.start()


t1.join()
# t2.join()

print("done")
