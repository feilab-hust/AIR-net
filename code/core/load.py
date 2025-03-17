import numpy as np
import imageio
import tifffile
import mat73
import json
import os



def load_data(path:str, datatype:str = 'LF', normalize:bool = True, ret_max_val:bool = False):
    if datatype == 'LF':
        images, H_mtxs = load_LF(path, normalize = normalize)
        out = images, H_mtxs
    else:
        raise NotImplementedError("Dataset type not supported!")
    
    return out

def load_LF(path:str, normalize:bool = True):
    """
    Load default LF data.
    path: basedir path
    normalize: normalize input images by their max value or not
    """
    with open(os.path.join(path, 'transforms_train.json'), 'r') as fp:
        meta = json.load(fp)
    if 'stack_path' in meta:
        imgs_list=[]
        for _f in  meta['stack_path']:
            stack_path = os.path.join(path, _f)
            print('loading viewstacks from %s' % stack_path)
            images, maxval = readtif(stack_path, normalize = normalize)
            images = images[...,None] if len(images.shape)==3 else images  # (N,H,W,C)
            imgs_list.append(images)
    else:
        raise IOError("Dataset not found!")
    H_centers, H_radius = None, None
    if 'H_cuts' in meta:
        H_centers = np.array(meta['H_cuts']['centers'], dtype=np.int32)
        H_radius = int(meta['H_cuts']['radius'])
    # get H matrice
    PSF_list=[]
    for _f in meta['H_path']:
        Hdir = os.path.join(path, _f)
        print('loading PSF from %s'%Hdir)
        H_mtxs = getH(Hdir, H_centers, H_radius)
        PSF_list.append(H_mtxs)
    N,H,W,C = images.shape
    Nm,D,Hm,Wm = H_mtxs.shape
    assert N == Nm ,"#images and #H_mtxs not matching!"
    return imgs_list, PSF_list

def readtif(path:str, normalize:bool = True, dtype = np.float32):
    """
    read tif or tiff image stack from file.
    Returns:
        out: image stack in ndarray; (N,H,W)
        maxval: max value in image, used for normalizing; float
    """
    out = tifffile.imread(path).astype(dtype)
    if len(out.shape) == 2:
        out = out[None,...]     # (N,H,W)
    # normalize
    if normalize:
        maxval = np.max(out)
        out /= maxval
    else:
        maxval = None
    return out, maxval

def getH(path:str, center=None, radius:int=None):
    if path.endswith('.mat'):
        mat = mat73.loadmat(path)
        for key in mat:
            raw = mat[key] # (H,W,D)
            break


        if center is not None and radius is not None:
            print(f"Read {raw.shape} integered raw H matrix from {path}")
            if len(raw.shape) != 4:
                center = np.array(center, dtype=np.int32)
                radius = int(radius)
                Hs = []
                for cc in center:
                    H = (raw[cc[0]-radius:cc[0]+radius+1,cc[1]-radius:cc[1]+radius+1,:]).astype(np.float32)
                    H = H.transpose(2,0,1)
                    Hs.append(H)
                H = np.array(Hs, dtype=np.float32)  # (N,D,H,W)
            else:
                center = np.array(center, dtype=np.int32)
                radius = int(radius)
                Hs = []
                for v_idx,cc in enumerate(center):
                    H = (raw[cc[0]-radius:cc[0]+radius+1,cc[1]-radius:cc[1]+radius+1,:,v_idx]).astype(np.float32)
                    H = H.transpose(2,0,1)
                    Hs.append(H)
                H = np.array(Hs, dtype=np.float32)  # (N,D,H,W)
        else:
            print(f"Read {raw.shape} H matrix from {path}")
            H = raw
    elif path.endswith('.npz') or path.endswith('.npy'):
        H = np.load(path)
    elif path.endswith(('.tif','.tiff')):
        H = tifffile.imread(path)
    else:
        raise ValueError("Not supported for this format, please use .tif/.tiff/.mat")
    H = H.astype(np.float32)
    print(f"H: {H.shape} {H.dtype}")
    return H