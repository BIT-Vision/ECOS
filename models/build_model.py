from config import cfg
import time
import torch
from .ECOSNet import build_ECOS

def build_model(args):
    model_type = cfg.MODEL.TYPE
    model = None
    if model_type == 'ECOS':
        model = build_ECOS(cfg)
    else:
        raise ValueError("Unsupport model type")
    return model