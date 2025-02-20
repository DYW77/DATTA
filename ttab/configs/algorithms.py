# -*- coding: utf-8 -*-

# 1. This file collects significant hyperparameters for the configuration of TTA methods.
# 2. We are only concerned about method-related hyperparameters here.
# 3. We provide default hyperparameters from the paper or official repo if users have no idea how to set up reasonable values.
import math

algorithm_defaults = {
    "no_adaptation": {"model_selection_method": "last_iterate"},
    "datta":{
        "optimizer": "SGD",# use Adam in the paper
        "memory_size": 64,
        "update_every_x": 64,  # This param may change in our codebase.
        "memory_type": "PBRS",
        "bn_momentum": 0.01,
        "temperature": 1.0,
        "iabn": False,  # replace bn with iabn layer
        "iabn_k": 4,
        "threshold_note": 1,  # skip threshold to discard adjustment.
        "use_learned_stats": True,
    },
    "tent": {
        "optimizer": "SGD",
    },
    "eata": {
        "optimizer": "SGD",
        "eata_margin_e0": math.log(1000)
        * 0.40,  # The threshold for reliable minimization in EATA.
        "eata_margin_d0": 0.05,  # for filtering redundant samples.
        "fishers": True, # whether to use fisher regularizer.
        "fisher_size": 2000,  # number of samples to compute fisher information matrix.
        "fisher_alpha": 50,  # the trade-off between entropy and regularization loss.
    },
    "note": {
        "optimizer": "SGD",  # use Adam in the paper
        "memory_size": 64,
        "update_every_x": 64,  # This param may change in our codebase.
        "memory_type": "PBRS",
        "bn_momentum": 0.01,
        "temperature": 1.0,
        "iabn": False,  # replace bn with iabn layer
        "iabn_k": 4,
        "threshold_note": 1,  # skip threshold to discard adjustment.
        "use_learned_stats": True,
    },
    "sar": {
        "optimizer": "SGD",
        "sar_margin_e0": math.log(1000)
        * 0.40,  # The threshold for reliable minimization in SAR.
        "reset_constant_em": 0.2,  # threshold e_m for model recovery scheme
    },
    "rotta":{
        "optimizer": "Adam",
        "nu": 0.001,
        "memory_size": 64,
        "update_frequency": 64,
        "lambda_t": 1.0,
        "lambda_u": 1.0,
        "alpha": 0.05,
    },
    "deyo": {
        "optimizer": "SGD",
        "filter_ent": True, # whether to filter samples by entropy
        "aug_type": "patch", # the augmentation type for prime
        "occlusion_size": 112, # choises for occ
        "row_start": 56, # choises for occ
        "column_start": 56, # choises for occ
        "patch_len": 4, # choises for patch
        "filter_plpd": True, # whether to filter samples by plpd
        "plpd_threshold": 0.3, # plpd threshold for DeYO
        "reweight_ent": 1, # reweight entropy loss
        "reweight_plpd": 1, # reweight plpd loss
    },
    "vida":{
        "optimizer": "SGD",
        'ViDALR': 1e-5,
        'WD': 0.,
        'MT': 0.999,
        'MT_ViDA': 0.999,
        'beta': 0.9,
        "vida_rank1": 1,
        "vida_rank2": 128,
        "unc_thr":0.2,
        "alpha_teacher": 0.999,
        "alpha_vida": 0.99,
        "bn_momentum": 0.01,
        "vida_rank1": 1,
        "vida_rank2": 128
        },
}
