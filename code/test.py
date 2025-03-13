import torch
import numpy as np
from tqdm import trange
import tifffile
import os

from core.load import load_data, getH
from core.models.Model import create_model,create_model_psf
from core.utils.LF import *
from core.utils.misc import *
from core.utils.Coord import Coord
from glbSettings import *
from core.utils.utils import *



def test(Flags):
    with torch.no_grad():
        ## Data preparation

        images, H_mtxs = load_data(path = Flags.datadir, datatype = Flags.datatype, normalize = True, ret_max_val = True)
        for idx, (_img, _h) in enumerate(zip(images, H_mtxs)):
            print(f'{Flags.datatype} dataset Loaded from {Flags.datadir}. Images: [%s]' % (str(_img[0].shape)),
                  'H_mtxs: ', _h.shape)

            N, H, W, C = images[idx].shape
            Nm, D, Hm, Wm = H_mtxs[idx].shape
            assert N == Nm, "The number of the images and the H matrice should be the same"
            assert (Hm % 2 == 1 and Wm % 2 == 1), "H matrix should have odd scale"
            H_mtxs[idx] = np.asarray([_h / np.sum(_h) for _h in H_mtxs[idx]])

        path = os.path.join(Flags.basedir, Flags.expname, 'renderonly')
        os.makedirs(os.path.join(path, 'recon'), exist_ok=True)
        os.makedirs(os.path.join(path, 'D1'), exist_ok=True)
        ## constants
        S = np.array([W,H,D], dtype=np.float32)
        maxHW = max(H,W)
        sc = np.full(3,2/(maxHW-1), dtype=np.float32)
        dc = -((S//2)*2)/(maxHW-1)
        # dc[1] *= -1     # flip Y
        # sc[1] *= -1
        Coord.set_idx2glb_scale(sc)
        Coord.set_idx2glb_trans(dc)

        ## Create model
        if Flags.weights_path is not None and Flags.weights_path!='None':
            ckpts = [Flags.weights_path]
        else:
            if Flags.aberration_correct:
                ckpts = [os.path.join(Flags.basedir, Flags.expname, f) for f in sorted(os.listdir(os.path.join(Flags.basedir, Flags.expname))) if 'AIR_cp_best' in f]
                print('Found ckpts', ckpts)
                if len(ckpts) > 0:
                    ckpt_path = ckpts[-1]
                    print('Reloading from', ckpt_path)
                else:
                    print("No ckpts found. Do nothing.")
                    return
                model, net_ker, embedder, post_processor, optimizer, start = create_model_psf(Flags, (D, H, W),
                                                                                              Flags.modeltype,
                                                                                              Flags.embeddertype,
                                                                                              Flags.postprocessortype,
                                                                                              lr=Flags.lrate,
                                                                                              weights_path=ckpt_path)
            else:
                ckpts = [os.path.join(Flags.basedir, Flags.expname, f) for f in sorted(os.listdir(os.path.join(Flags.basedir, Flags.expname))) if 'INR_cp_best'  in f]
                print('Found ckpts', ckpts)
                if len(ckpts) > 0:
                    ckpt_path = ckpts[-1]
                    print('Reloading from', ckpt_path)
                else:
                    print("No ckpts found. Do nothing.")
                    return
                model, embedder, post_processor, optimizer, start = create_model(Flags, (D, H, W), Flags.modeltype,
                                                                                 Flags.embeddertype,
                                                                                 Flags.postprocessortype,
                                                                                 lr=Flags.lrate, weights_path=ckpt_path)




        ## Create grid indice
        idx_glb_all = torch.stack(torch.meshgrid(
                                    torch.arange(D),
                                    torch.arange(H),
                                    torch.arange(W), 
                                    indexing = 'ij'), axis=-1)  # (D,H,W,3)
        pts_glb_all = Coord.idx2glb(idx_glb_all)
        ## to GPU
        H_mtxs = torch.Tensor(H_mtxs).to(DEVICE)
        pts_glb_all = pts_glb_all.to(DEVICE)

        voxel = get_block(pts_glb_all, model, embedder, post_processor, chunk=Flags.chunk)
        pred1 = torch.stack([forward_project(voxel, H_mtx) for H_mtx in H_mtxs[0]])
        filename = 'aberra_ViewStack.tif' if Flags.aberration_correct else 'ViewStack.tif'
        tifffile.imwrite(os.path.join(path, 'D1', filename ),
                         np.squeeze(pred1.cpu().numpy()))
        filename = 'aberra_Rendering_best.tif' if Flags.aberration_correct else 'Rendering_best.tif'
        tifffile.imwrite(
            os.path.join(path, 'recon', filename),
            np.squeeze(voxel.cpu().numpy()))
        visualize_3d_images(voxel.cpu().numpy(), Flags)
