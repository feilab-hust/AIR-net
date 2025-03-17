import torch
import numpy as np



def forward_project(realspace:torch.Tensor, H_mtx:torch.Tensor, padding:str='same'):
    """
    Perform forward projection.
    realspace: object space volume voxel
    H_mtx: H matrix used for forward projection. (Related to PSF)
    padding: padding added to each dim and each side. See torch.nn.functional.pad().
    """
    D,H,W,C = realspace.shape
    Dm,Hm,Wm = H_mtx.shape
    assert D == Dm, "The Depth of realspace and H_matrix is not matched!"
    realspace_padded = realspace
    try:
        realspace_padded = torch.nn.functional.pad(realspace_padded, pad=padding, value=0)
        padding = 'valid'
    except: pass
    # out = torch.nn.functional.conv2d(realspace_padded, H_mtx[None,...], None, padding='valid')
    realspace_perm = realspace_padded.permute([-1,0,1,2])   # (C,D,H,W)
    H_perm = H_mtx[None,...]                                # (1,Dm,Hm,Wm)
    out = torch.nn.functional.conv2d(realspace_perm, H_perm, None, padding=padding).permute([1,2,3,0])[0]   # (H,W,C)
    return out

def DeConv(projections:torch.Tensor, Hts:torch.Tensor, weights:torch.Tensor or int = 1):
    """
    Perform back projection, by deconvolution.
    """
    N,H,W,C = projections.shape
    Nm,Dm,Hm,Wm = Hts.shape
    assert N == Nm, "#Hts and #projections is not matched!"
    weights = torch.tensor([weights/N], device=projections.device)
    projections_perm = (projections * weights.reshape([-1,1,1,1])).permute([3,0,1,2]) # (C,N,H,W)
    Hts_perm = Hts.permute([1,0,2,3])   # (D,N,H,W)
    realspace = torch.nn.functional.conv2d(projections_perm, Hts_perm, None, padding='same')    # (C,D,H,W)
    realspace = realspace.permute([1,2,3,0])    # (D,H,W,C)
    return realspace



if __name__=='__main__':
    import tifffile
    import os
    import json
    from tqdm import trange
    from glbSettings import *
    from core.load import getH


    print("Run rendering on CUDA: ", torch.cuda.is_available())
    torch.set_default_tensor_type('torch.cuda.FloatTensor') if torch.cuda.is_available() else torch.set_default_tensor_type("torch.FloatTensor")
    with torch.no_grad():
        # O_dir = "E:\INR_datas\GT_stack.tif"
        O_basedir = r"J:\Nerf_test\Data\20230426_clb_tube\registration\Seq_1\whole_fov\scale_padding\paired\stack\2023-06-01_Paddding_Znum_5"
        H_dir = r"E:\NetCode\Nerf_test\LF_INR_MixedFusion\data\LF\view7_roi2\psf\View7_full_mat.mat"
        out_dir = "./out"

        centercrops = [[1017,1200],[1017,1700],[1450,951],[1450,1450],[1450,1949],[1883,1200],[1883,1700]]
        radius = 81//2

        Hs = getH(H_dir, center=centercrops, radius=radius)
        Hs = np.array([H / np.sum(H) for H in Hs], dtype=np.float32)
        Hs = torch.from_numpy(Hs).to(DEVICE)    # (D,H,W)

        metas = {"scenes":[]}
        os.makedirs(os.path.join(O_basedir, 'train'), exist_ok=True)
        for f in sorted(os.listdir(O_basedir)):
            if '.tif' in f:
                O_dir = os.path.join(O_basedir, f)
                O = np.array(tifffile.imread(O_dir)[::-1,...,None], dtype=np.float32)
                O = torch.from_numpy(O).to(DEVICE)  # (D,H,W)
                outs = []
                for i in trange(Hs.shape[0]):
                    out = forward_project(O, Hs[i])
                    outs.append(out)
                outs = torch.stack(outs, dim=0)
                outname = f'{os.path.splitext(f)[0]}_proj.tif'
                tifffile.imwrite(os.path.join(O_basedir, 'train', outname), outs.to("cpu").numpy())

                meta = {
                        "targets_path": f"./train/{outname}",
                        "views_path": f"./train/{outname}",
                        "H_path": "./train/View7_full_mat.mat",
                        "H_cuts": {
                            "radius": 40,
                            "centers":[[1017,1200],[1017,1700],[1450,951],[1450,1450],[1450,1949],[1883,1200],[1883,1700]]
                            }
                        }
                metas["scenes"].append(meta)
        metas_js = json.dumps(metas, indent=4)
        with open(os.path.join(O_basedir, "transforms_train.json"), "w") as f:
            f.write(metas_js)

        # back_project test
        # outs = np.array(tifffile.imread('E:\INR_Projs\LF-INR-mvs_block\data\LF-MS\LF_view7_LF-MS_1_ROI_opt\\train\customed_psf_projection.tif'), dtype=np.float32)
        # outs = torch.from_numpy(outs).to(DEVICE)[...,None]
        # Hts = getHt('E:\INR_Projs\LF-INR-mvs_block\data\LF-MS\LF_view7_LF-MS_1_ROI_opt\\train\\new_psf.npy')
        # Hts = torch.from_numpy(Hts).to(DEVICE)
        # Hts = torch.tensor(getHt(H_dir, center=centercrops, radius=radius), device=DEVICE)
        # Ohat = DeConv(outs, Hts).to("cpu").numpy()
        # tifffile.imwrite(out_dir+'_bp.tif', Ohat[::-1,...,0])