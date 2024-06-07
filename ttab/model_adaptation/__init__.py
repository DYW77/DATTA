# -*- coding: utf-8 -*-
from .bn_adapt import BNAdapt
from .conjugate_pl import ConjugatePL
from .cotta import CoTTA
from .eata import EATA
from .memo import MEMO
from .no_adaptation import NoAdaptation
from .note import NOTE
from .sar import SAR
from .shot import SHOT
from .t3a import T3A
from .tent import TENT
from .ttt import TTT
from .ttt_plus_plus import TTTPlusPlus
from .rotta import Rotta
from .deyo import DEYO
from .dyn import DYN
from .tent_iabn import TENT_iabn
from .vida import ViDA
def get_model_adaptation_method(adaptation_name):
    return {
        "no_adaptation": NoAdaptation,
        "tent": TENT,
        "bn_adapt": BNAdapt,
        "memo": MEMO,
        "shot": SHOT,
        "t3a": T3A,
        "ttt": TTT,
        "ttt_plus_plus": TTTPlusPlus,
        "note": NOTE,
        "sar": SAR,
        "conjugate_pl": ConjugatePL,
        "cotta": CoTTA,
        "eata": EATA,
        "rotta": Rotta,
        "deyo":DEYO,
        "dyn":DYN,
        "tent_iabn":TENT_iabn,
        "vida":ViDA,
    }[adaptation_name]
