import mat73
import torch
import numpy as np
from torchvision.transforms.functional import rotate
from torchvision.transforms import InterpolationMode


# def Ortho_Rot_prj(realspace:torch.Tensor, H_mtx:torch.Tensor, padding:str='same'):
#     """
#     Perform forward projection.
#     realspace: object space volume voxel
#     H_mtx: H matrix used for forward projection. (Related to PSF)
#     padding: padding added to each dim and each side. See torch.nn.functional.pad().
#     """
#     D,H,W,C = realspace.shape
#     Dm,Hm,Wm = H_mtx.shape
#     assert D == Dm, "The Depth of realspace and H_matrix is not matched!"
#     realspace_padded = realspace
#     try:
#         realspace_padded = torch.nn.functional.pad(realspace_padded, pad=padding, value=0)
#         padding = 'valid'
#     except: pass
#     # out = torch.nn.functional.conv2d(realspace_padded, H_mtx[None,...], None, padding='valid')
#     realspace_perm = realspace_padded.permute([-1,0,1,2])   # (C,D,H,W)
#     H_perm = H_mtx[None,...]                                # (1,Dm,Hm,Wm)
#     # out = torch.nn.functional.conv2d(realspace_perm, H_perm, None, padding=padding).permute([1,2,3,0])[0]   # (H,W,C)
#     out = fft_conv(realspace_perm, H_perm, None, padding=padding).permute([1, 2, 3, 0])[0]  # (H,W,C)
#     return out



def Ortho_Rot_prj(X, axis, theta, expand=False, fill=0.0):
    '''
    input X: (depth,height,width)
    angle: rotate angle
    '''

    if axis == 0:
        X = rotate(X, interpolation=InterpolationMode.BILINEAR, angle=theta, expand=expand, fill=fill)
    elif axis == 1:
        X = X.permute((1, 0, 2))
        X = rotate(X, interpolation=InterpolationMode.BILINEAR, angle=theta, expand=expand, fill=fill)
        X = X.permute((1, 0, 2))
    elif axis == 2:
        X = X.permute((2, 1, 0))
        X = rotate(X, interpolation=InterpolationMode.BILINEAR, angle=-theta, expand=expand, fill=fill)
        X = X.permute((2, 1, 0))
    else:
        raise Exception('Not invalid axis')

    prj = torch.sum(X.squeeze(0),dim=0)
    return prj





if __name__=='__main__':
    import tifffile
    from glbSettings import *
    import os
    from utils import *
    print("Run rendering on CUDA: ", torch.cuda.is_available())
    torch.set_default_tensor_type('torch.cuda.FloatTensor') if torch.cuda.is_available() else torch.set_default_tensor_type("torch.FloatTensor")

    angle_range = np.asarray([-52, 52])
    # angle_range = angle_range * np.pi / 180
    angle_num = 5
    angle_interval = (angle_range[1] - angle_range[0]) / (angle_num - 1)
    angle_list = np.arange(angle_range[0], angle_range[1] + 0.001, angle_interval)

    O_dir = r"E:\Matlab_Code\Degradation_Code\ViewSyn\ZFH_Fiber\20240217\vol\23-Feb-2024_Paddding_Znum_1"
    save_path = O_dir
    out_dir = os.path.join(save_path,'Pytorch_Synthetic_projection_num%s/'%angle_num)
    exists_or_mkdir(out_dir)
    img_list = load_imgs(O_dir)

    for img_idx, _f in enumerate(img_list):
        vol = tifffile.imread(os.path.join(O_dir,_f))
        print('[%d/%d] Loading %s -- Size: %s'%(img_idx,len(img_list),_f, str(vol.shape) ))
        vol = np.asarray(vol,np.float32)
        vol = normalize(vol)
        real_space = torch.from_numpy(vol).to(DEVICE)
        prj_list = torch.zeros([angle_num,*vol.shape[1:]]).to(DEVICE)
        for _ang in angle_list:
            rg_vol=Ortho_Rot_prj(real_space,axis=1,theta=_ang)
            prj_list =
            pass

