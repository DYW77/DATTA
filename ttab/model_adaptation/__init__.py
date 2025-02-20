# -*- coding: utf-8 -*-
from .no_adaptation import NoAdaptation
from .note import NOTE
from .sar import SAR
from .tent import TENT
from .rotta import Rotta
from .deyo import DEYO
from .vida import ViDA
from .datta import DATTA
def get_model_adaptation_method(adaptation_name):
    return {
        "no_adaptation": NoAdaptation,
        "tent": TENT,
        "note": NOTE,
        "sar": SAR,
        "rotta": Rotta,
        "deyo":DEYO,
        "vida":ViDA,
        "datta":DATTA,
    }[adaptation_name]
