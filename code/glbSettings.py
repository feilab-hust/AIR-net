import torch
##use cuda
GPU_IDX = 0
DEVICE = torch.device(f"cuda:{GPU_IDX}" if torch.cuda.is_available() else "cpu")

torch.cuda.set_device(DEVICE)

##use cpu
# DEVICE = torch.device("cpu")
