# -*- coding: utf-8 -*-
from .bn_adapt import BNAdapt
from .eata import EATA
from .no_adaptation import NoAdaptation
from .note import NOTE
from .sar import SAR
from .tent import TENT
from .rotta import Rotta
from .deyo import DEYO
from .dyn import DYN
from .iabn import IABN
from .vida import ViDA
def get_model_adaptation_method(adaptation_name):
    return {
        "no_adaptation": NoAdaptation,
        "tent": TENT,
        "bn_adapt": BNAdapt,
        "note": NOTE,
        "sar": SAR,
        "eata": EATA,
        "rotta": Rotta,
        "deyo":DEYO,
        "dyn":DYN,
        "iabn":IABN,
        "vida":ViDA,
    }[adaptation_name]
