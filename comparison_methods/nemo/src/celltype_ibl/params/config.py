import os

C4_COLORS = {
    "PkC_ss": [28, 120, 181],
    "PkC_cs": [0, 0, 0],
    "MLI": [224, 85, 159],
    "MFB": [214, 37, 41],
    "GrC": [143, 103, 169],
    "GoC": [56, 174, 62],
    "laser": [96, 201, 223],
    "drug": [239, 126, 34],
    "background": [244, 242, 241],
    "MLI_A": [224, 85, 150],
    "MLI_B": [220, 80, 160],
}

REGION_COLORS = {
    "CB": [240, 240, 128],
    "CNU": [152, 214, 249],
    "CTXsp": [138, 218, 135],
    "HB": [255, 155, 136],
    "HPF": [126, 208, 75],
    "HY": [230, 68, 56],
    "Isocortex": [112, 255, 113],
    "MB": [255, 100, 255],
    "OLF": [154, 210, 189],
    "TH": [255, 112, 128],
}

V1_COLORS = {
    "PV": [214, 37, 41],
    "SST": [143, 103, 169],
    "VIP": [56, 174, 62],
}


WVF_ENCODER_ARGS_SINGLE = {
    "beta": 5,
    "d_latent": 10,
    "dropout_l0": 0.1,
    "dropout_l1": 0.1,
    "lr": 1e-4,
    "n_layers": 2,
    "n_units_l0": 600,
    "n_units_l1": 300,
    "optimizer": "Adam",
    "batch_size": 128,
}

# directories for ibl models - set to None to disable loading pretrained models
CLIP_MODEL_DIR = None
VAE_DIR = None
SIMCLR_ACG_DIR = None
SIMCLR_WVF_DIR = None

# wandb directotry
# WANDB_DIR = "/mnt/sdceph/users/hyu10/cell-type_representation/wandb_save"
# WANDB_PROJECT = "ibl"
# WANDB_ENTITY = "contrastive_neuron"
WANDB_ENTITY="mostajo-group"
WANDB_PROJECT="celltype-ibl"
WANDB_DIR = os.environ.get("NEMO_WANDB_DIR", "./wandb_save")
# c4, ibl data directories
DATASETS_DIRECTORY = os.environ.get("NEMO_DATASETS_DIR", "./datasets")


# Dataset paths - will be looked for in DATASETS_DIRECTORY
ALLEN_LABELLED_PATH = None  # Set to filename if available in datasets directory
ALLEN_UNLABELLED_PATH = None  # Set to filename if available in datasets directory

CELL_EXPLORER_PATH = None  # Set to filename if available in datasets directory

SPLIT_INDEX_PATH = None  # Set to directory name if available in datasets directory
INDEX_MAPPING_PATH = None  # Set to filename if available in datasets directory
KENJI_ALLEN_PATH = None  # Set to filename if available in datasets directory

# ultra data directories
ULTRA_DATA_PATH = None  # Set to filename if available in datasets directory
STIM_REMOVED_DATA_PATH = None  # Set to filename if available in datasets directory
