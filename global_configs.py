import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

DEVICE = torch.device("cuda:0")

DATASET_CONFIGS = {
    "mosi": {
        "ACOUSTIC_DIM": 74,
        "VISUAL_DIM": 47,
        "TEXT_DIM": 768
    },
    "mosei": {
        "ACOUSTIC_DIM": 74,
        "VISUAL_DIM": 35,
        "TEXT_DIM": 768
    }
}