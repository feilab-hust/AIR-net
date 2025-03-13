"""
Create positional encoding embedder
"""
import torch

from glbSettings import *


class Embedder:
    @staticmethod
    def get_by_name(embedder_type:str=None, *args, **kwargs):
        if embedder_type is None or embedder_type == 'None':
            embedder_type = "BasicEmbedder"
        try:
            embedder = eval(embedder_type)(*args, **kwargs)
        except:
            raise ValueError(f"Type {embedder_type} not recognized!")
        return embedder

class BasicEmbedder:
    """
    An embedder that do nothing to the input tensor.
    Used as a template.
    """
    def __init__(self, *args, **kwargs):
        """To be overwrited"""
        del args, kwargs
    
    def embed(self, inputs):
        """To be overwrited"""
        return inputs

class PositionalEncoder(BasicEmbedder):
    def __init__(self, multires=0, multiresZ = None, input_dim=3, embed_bases=[torch.sin,torch.cos], include_input=True, *args, **kwargs):
        del args, kwargs
        self.multires = multires
        self.multiresZ = multiresZ if multiresZ is not None else multires,
        self.in_dim = input_dim
        self.embed_bases = embed_bases
        self.include_input = include_input
        self.embed_fns = []
        self.out_dim = 0
        self._create_embed_fn()

    def _create_embed_fn(self):
        embed_fns = []
        d = self.in_dim
        out_dim = 0
        if self.include_input:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.multires-1
        N_freqs = self.multires
        freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs).to(DEVICE)
            
        for freq in freq_bands:
            for p_fn in self.embed_bases:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)
