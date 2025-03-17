import numpy as np
import torch
from core.utils.pytorch_ssim import SSIM
from glbSettings import *


ssim_loss= SSIM(window_size=11)
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]).to(DEVICE))

def get_block(pts, model, embedder, post=torch.relu, chunk=32*1024, model_kwargs={'embedder':{}, 'post':{}}):
    sh = pts.shape[:-1]
    pts = pts.reshape([-1,pts.shape[-1]])
    # break down into small chunk to avoid OOM
    outs = []
    for i in range(0, pts.shape[0], chunk):
        pts_chunk = pts[i:i+chunk]
        # eval
        out = model.eval(pts_chunk, embedder, post, model_kwargs)
        outs.append(out)
    outs = torch.cat(outs, dim=0)
    outs = outs.reshape([*sh,-1])
    return outs

def edge_loss(pred,target):
    kernels = [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
               [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]
    # padding=  [[0, 0], [1, 1], [1, 1], [0, 0]]
    kernel_x=torch.from_numpy(np.asarray(kernels[0],np.float32)).to(DEVICE)
    kernel_y = torch.from_numpy(np.asarray(kernels[1], np.float32)).to(DEVICE)
    # kernelsX_rep=torch.expand_copy()
    # kernelsY_rep =
    # # pred_paddi
    pred_edge_x = torch.nn.functional.conv2d(input=pred[None,None,...,0], weight=kernel_x.reshape(-1,1,3,3),padding=1)
    pred_edge_y = torch.nn.functional.conv2d(input=pred[None,None,...,0], weight=kernel_y.reshape(-1,1,3,3),padding=1)

    target_edge_x= torch.nn.functional.conv2d(input=target[None,None,...,0], weight=kernel_x.reshape(-1,1,3,3),padding=1)
    target_edge_y = torch.nn.functional.conv2d(input=target[None, None, ..., 0], weight=kernel_y.reshape(-1, 1, 3, 3),
                                               padding=1)

    return (img2mse(pred_edge_x,target_edge_x)+img2mse(pred_edge_y,target_edge_y))/2