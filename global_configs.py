import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

DEVICE = torch.device("cuda:0")

# MOSI SETTING
TEXT_DIM = 768  # x0
ACOUSTIC_DIM = 74  # x1
VISUAL_DIM = 47  # x2


# # MOSEI SETTING
# ACOUSTIC_DIM = 74
# VISUAL_DIM = 35
# TEXT_DIM = 768
