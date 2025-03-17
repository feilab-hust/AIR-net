import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.models.Embedder import *
from core.utils.Coord import Coord
from glbSettings import *

dtype = torch.cuda.FloatTensor
class Model:
    def __init__(self, model_type:str, *args, **kwargs):
        self.type = model_type
        self.model = self.get_by_name(model_type, *args, **kwargs)

    def get_by_name(self, model_type:str, *args, **kwargs):
        try:
            model = eval(model_type)(*args, **kwargs)
        except:
            raise ValueError(f"Type {model_type} not recognized!")
        return model

    def eval(self, x, embedder, post, model_kwargs):
        embedder_kwargs = model_kwargs['embedder']
        post_kwargs = model_kwargs['post']
        h = x
        h = embedder.embed(h, **embedder_kwargs)
        h = self.model(h)
        h = post(h, **post_kwargs)
        return h
    
    def get_model(self):
        return self.model

    def get_state(self):
        return self.model.get_state()

    def load(self, ckpt):
        self.model.load_params(ckpt)

class BasicModel(nn.Module):
    """Basic template for creating models."""
    def __init__(self):
        super().__init__()

    def forward(self):
        """To be overwrited"""
        raise NotImplementedError
    
    def load_params(self, path):
        """To be overwrited"""
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError

class NeRF(BasicModel):
    """Standard NeRF model"""
    def __init__(self, D=8, W=256, input_ch=3, output_ch=1, skips=[4], *args, **kwargs):
        del args, kwargs
        super().__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.skips = skips

        layers = [nn.Linear(input_ch, W)]
        for i in range(D-1):
            in_channels = W
            if i in self.skips:
                in_channels += input_ch
            layers.append(nn.Linear(in_channels, W))
        self.pts_linear = nn.ModuleList(layers)
        self.output_linear = nn.Linear(W, output_ch)
        self.to(DEVICE)
        
    def forward(self, x):
        h = x
        for i,l in enumerate(self.pts_linear):
            h = l(h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x, h], dim=-1)
        outputs = self.output_linear(h)
        return outputs
    
    def load_params(self, ckpt:dict):
        # Load model
        self.load_state_dict(ckpt['network_fn_state_dict'])

    def get_state(self):
        return self.state_dict() 



class vap_module(nn.Module):
    def __init__(self, max_val=0.5, init_value=None, zernike_terms_num=5,Nm=9):
        super(vap_module, self).__init__()

        if init_value == None:

            _k4 = torch.rand((Nm,1), device=f'cuda:{GPU_IDX}', requires_grad=True)

            _k5 = torch.full((Nm,1), fill_value=0.5, device=f'cuda:{GPU_IDX}', requires_grad=True)

            if zernike_terms_num == 4:
                _k6_15 = torch.rand((Nm,10), device=f'cuda:{GPU_IDX}', requires_grad=True)
                _k0 = 2 * max_val * torch.cat((_k4, _k5, _k6_15), 1) - max_val
                self.k = torch.nn.Parameter(_k0).type(dtype)

            elif zernike_terms_num == 5:
                _k6_21 = torch.rand((Nm,16), device=f'cuda:{GPU_IDX}', requires_grad=True)

                self.k = torch.nn.Parameter(2 * max_val * torch.cat((_k4, _k5, _k6_21), 1) - max_val).type(dtype)

            elif zernike_terms_num == 6:
                _k6_28 = torch.rand((Nm,23), device=f'cuda:{GPU_IDX}', requires_grad=True)

                self.k = torch.nn.Parameter(2 * max_val * torch.cat((_k4, _k5, _k6_28), 1) - max_val).type(dtype)

            elif zernike_terms_num == 7:
                _k6_36 = torch.rand((Nm,30), device=f'cuda:{GPU_IDX}', requires_grad=True)

                self.k = torch.nn.Parameter(2 * max_val * torch.cat((_k4, _k5, _k6_36), 1) - max_val).type(dtype)

            elif zernike_terms_num == 8:
                _k6_45 = torch.rand(((Nm,40)), device=f'cuda:{GPU_IDX}', requires_grad=True)

                self.k = torch.nn.Parameter(2 * max_val * torch.cat((_k4, _k5, _k6_45), 1) - max_val).type(dtype)

            elif zernike_terms_num == 9:
                _k6_55 = torch.rand((Nm,50), device=f'cuda:{GPU_IDX}', requires_grad=True)

                self.k = torch.nn.Parameter(2 * max_val * torch.cat((_k4, _k5, _k6_55), 1) - max_val).type(dtype)

        else:
            self.k = torch.nn.Parameter(torch.tensor(init_value, device=f'cuda:{GPU_IDX}', requires_grad=False)).type(dtype)

    def forward(self):
        return self.k

def create_model(Flags, shape, model_type='NeRF', embedder_type='PositionalEncoder', post_processor_type='relu', lr=1e-4, weights_path=None):
    ## embedder
    embedder_kwargs = {
        # PositionalEmbedder
        'multires': Flags.multires, 
        'multiresZ': Flags.multiresZ,
        'input_dim': 3
    }
    embedder = Embedder.get_by_name(embedder_type, **embedder_kwargs)
    ## model
    model_kwargs={
                # NeRF
                'D': Flags.netdepth,
                'W': Flags.netwidth,
                'input_ch': embedder.out_dim if hasattr(embedder, 'out_dim') else 3,
                'output_ch': Flags.sigch,
                'skips': Flags.skips, 
                # Grid
                'shape': (*shape,1), 
                # Grid_NeRF
                'grid_shape':Flags.grid_shape,
                'overlapping': Flags.overlapping
                }
    model = Model(model_type=model_type, **model_kwargs)
    grad_vars = list(model.get_model().parameters())
    optimizer = torch.optim.Adam(params=grad_vars, lr=lr, betas=(0.9,0.999), capturable=(DEVICE.type == 'cuda'))
    start = 0
    # Load checkpoint
    if weights_path != None:
        ckpt = torch.load(weights_path, map_location=DEVICE)
        model.load(ckpt)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start = ckpt['global_step']
    ## post processor
    if post_processor_type == 'linear':
        post_processor = lambda x:x
    elif post_processor_type == 'relu':
        post_processor = torch.relu
    elif post_processor_type == 'leakrelu':
        post_processor = F.leaky_relu
    else:
        raise ValueError(f"Type {post_processor_type} not recognized!")
    return model, embedder, post_processor, optimizer, start
def create_model_psf(Flags, shape, model_type='NeRF', embedder_type='PositionalEncoder', post_processor_type='relu', lr=1e-4, weights_path=None):
    ## embedder
    embedder_kwargs = {
        # PositionalEmbedder
        'multires': Flags.multires,
        'multiresZ': Flags.multiresZ,
        'input_dim': 3
    }
    embedder = Embedder.get_by_name(embedder_type, **embedder_kwargs)
    ## model
    model_kwargs={
                # NeRF
                'D': Flags.netdepth,
                'W': Flags.netwidth,
                'input_ch': embedder.out_dim if hasattr(embedder, 'out_dim') else 3,
                'output_ch': Flags.sigch,
                'skips': Flags.skips,
                # Grid
                'shape': (*shape,1),
                # Grid_NeRF
                'grid_shape':Flags.grid_shape,
                'overlapping': Flags.overlapping
                }
    model = Model(model_type=model_type, **model_kwargs)
    grad_vars = list(model.get_model().parameters())
    net_ker = vap_module(max_val=Flags.zernike_max_val, zernike_terms_num=Flags.zernike_terms_num,Nm=Flags.Nnum)  # 5e-2

    optimizer = torch.optim.Adam([{'params': model.get_model().parameters(), 'lr': lr},  # 1e-3
                                  {'params': net_ker.parameters(), 'lr': Flags.lrate_kenel}],  # 4e-3
                                 betas=(0.9, 0.999), eps=1e-8)
    start = 0
    # Load checkpoint
    if weights_path != None:
        ckpt = torch.load(weights_path, map_location=DEVICE)
        model.load(ckpt)
        # optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start = ckpt['global_step']
    ## post processor
    if post_processor_type == 'linear':
        post_processor = lambda x:x
    elif post_processor_type == 'relu':
        post_processor = torch.relu
    elif post_processor_type == 'leakrelu':
        post_processor = F.leaky_relu
    else:
        raise ValueError(f"Type {post_processor_type} not recognized!")
    return model, net_ker,embedder, post_processor, optimizer, start#,kernel_optimizer
if __name__ == '__main__':
    model = NeRF()
    print(model.parameters)