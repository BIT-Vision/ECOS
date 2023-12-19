from easydict import EasyDict as edict
import yaml
"""
Add default config for ECOSNet.
"""
cfg = edict()

# MODEL
cfg.MODEL = edict()
cfg.MODEL.TYPE = 'ECOS'

cfg.MODEL.INPUT_DIM = 5
cfg.MODEL.NORM = 'GN' # BN GN
cfg.MODEL.HIDDEN_DIM = 256

# TRAIN
cfg.TRAIN = edict()

cfg.TRAIN.LR = 2e-4
cfg.TRAIN.WEIGHT_DECAY = 0.07

cfg.TRAIN.ITER = 40000
cfg.TRAIN.BATCH_SIZE = 1
cfg.TRAIN.NUM_WORKER = 4

cfg.TRAIN.SKIP_SAVE = 5000

cfg.DATA = edict()
cfg.DATA.ROOT = '../../../EventData'
cfg.DATA.SIZE =  (384, 384)

# DATA.TRAIN
cfg.DATA.TRAIN = edict()
cfg.DATA.TRAIN.DATA_MODE = 'event_5'
cfg.DATA.TRAIN.LABEL_MODE = 'event_label_format'

cfg.DATA.TRAIN.FRAMES_PER_CLIP = 20
cfg.DATA.TRAIN.MAX_OBJECTS = 4

# DATA.VAL
cfg.DATA.VAL = edict()
cfg.DATA.VAL.DATASET_NAME = "EOS"


def _edict2dict(dest_dict, src_edict):
    if isinstance(dest_dict, dict) and isinstance(src_edict, dict):
        for k, v in src_edict.items():
            if not isinstance(v, edict):
                dest_dict[k] = v
            else:
                dest_dict[k] = {}
                _edict2dict(dest_dict[k], v)
    else:
        return


def gen_config(config_file):
    cfg_dict = {}
    _edict2dict(cfg_dict, cfg)
    with open(config_file, 'w') as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)


def _update_config(base_cfg, exp_cfg):
    if isinstance(base_cfg, dict) and isinstance(exp_cfg, edict):
        for k, v in exp_cfg.items():
            if k in base_cfg:
                if not isinstance(v, dict):
                    base_cfg[k] = v
                else:
                    _update_config(base_cfg[k], v)
            else:
                raise ValueError("{} not exist in config.py".format(k))
    else:
        return


def update_config_from_file(filename):
    exp_config = None
    with open(filename) as f:
        exp_config = edict(yaml.safe_load(f))
        _update_config(cfg, exp_config)
