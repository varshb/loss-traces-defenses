import os


STORAGE_DIR = "./trained_models" # storing all trained models and MIA results
DATA_DIR = "./trained_models/datasets" # where to downloads training datasets
CINIC_10_PATH = "./cinic10" # for CINIC-10 we expect it to be pre-downloaded


# Can be overridden by creating a config_local.py file
try:
    from loss_traces.config_local import *  # noqa: F403
except ImportError:
    pass


MODEL_DIR = os.path.join(STORAGE_DIR, "models")