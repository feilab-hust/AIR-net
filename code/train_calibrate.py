import torch
import numpy as np
from tqdm import tqdm, trange
import tifffile
import os

from core.load import load_data
from core.models.Model import create_model,create_model_psf
from core.utils.LF import *
from core.utils.misc import *

from core.utils.utils import *
from core.utils.Coord import Coord
from glbSettings import *
from torch.utils.tensorboard import SummaryWriter
from psf_gen_abre import PsfGenerator

def  train(Flags):
    ## Load data from file
    print('ExpName: %s'%Flags.expname)


    images, H_mtxs = load_data(path = Flags.datadir, datatype = Flags.datatype, normalize = True, ret_max_val = True)

    for idx,(_img,_h) in enumerate(zip(images,H_mtxs)):
        print(f'{Flags.datatype} dataset Loaded from {Flags.datadir}. Images: [%s]'%(str(_img.shape)), 'H_mtxs: ', _h.shape)

        N, H, W, C = images[idx].shape
        Nm, D, Hm, Wm = H_mtxs[idx].shape
        assert N == Nm, "The number of the images and the H matrice should be the same"
        assert (Hm % 2 == 1 and Wm % 2 == 1), "H matrix should have odd scale"
        H_mtxs[idx] = np.asarray([_h/np.sum(_h) for _h in H_mtxs[idx]])

    br = Flags.block_size//2
    S = np.array([W,H,D], dtype=np.float32)
    maxHW = max(H,W)
    sc = np.full(3,2/(maxHW-1), dtype=np.float32)
    dc = -((S//2)*2)/(maxHW-1)
    # dc[1] *= -1     # flip Y
    # sc[1] *= -1
    # dc[2] *= -1     # flip Z
    # sc[2] *= -1
    Coord.set_idx2glb_scale(sc)
    Coord.set_idx2glb_trans(dc)

    ## Save config
    os.makedirs(os.path.join(Flags.basedir, Flags.expname), exist_ok=True)
    os.makedirs(os.path.join(Flags.basedir, Flags.expname,'D1'), exist_ok=True)
    os.makedirs(os.path.join(Flags.basedir, Flags.expname, 'D2'), exist_ok=True)
    os.makedirs(os.path.join(Flags.basedir, Flags.expname, 'recon'), exist_ok=True)

    Flags.append_flags_into_file(os.path.join(Flags.basedir, Flags.expname, 'config.cfg'))  # @@@



    ## Create grid indice
    idx_glb_all = torch.stack(torch.meshgrid(
                                torch.arange(D),
                                torch.arange(H),
                                torch.arange(W),
                                indexing = 'ij'), axis=-1)  # (D,H,W,3)
    pts_glb_all = Coord.idx2glb(idx_glb_all)
    ## to GPU
    images = torch.Tensor(images).to(DEVICE)
    H_mtxs = torch.Tensor(H_mtxs).to(DEVICE)
    pts_glb_all = pts_glb_all.to(DEVICE)
    writer = SummaryWriter(log_dir='./logs/%s/Tensorboard' % Flags.expname)
    #################################
    ##          CORE LOOP          ##
    #################################
    ## run
    N_steps = Flags.N_steps+1
    print("Start training...")
    start = 0

    #
    if not Flags.aberration_correct:
        ## Create model
        if Flags.weights_path is not None and Flags.weights_path != 'None':
            ckpts = [Flags.weights_path]
        else:
            ckpts = [os.path.join(Flags.basedir, Flags.expname, f) for f in
                     sorted(os.listdir(os.path.join(Flags.basedir, Flags.expname))) if 'INR_cp_best'  in f]

        print('Found ckpts', ckpts)
        if len(ckpts) > 0 and not Flags.no_reload:
            ckpt_path = ckpts[-1]
            print('Reloading from', ckpt_path)
        else:
            ckpt_path = None
            print("Creating new model...")

        model,embedder, post_processor, optimizer, start = create_model(Flags, (D, H, W), Flags.modeltype,
                                                                         Flags.embeddertype, Flags.postprocessortype,
                                                                         lr=Flags.lrate, weights_path=ckpt_path)
        # post processor
        if (post_processor is torch.relu or post_processor is torch.relu_) and start < 500:
            post_processor_old, post_processor = post_processor, torch.nn.functional.leaky_relu
        start += 1
        psnr=0
        for step in trange(start, N_steps):
            #
            # get random block center (c_idx,c_idy)
            c_idx = np.random.randint(br, W - br)
            c_idy = np.random.randint(br, H - br)

            # c_idx = images.shape[3]//2
            # c_idy = images.shape[2]//2

            # views_permute=np.random.permtation(N)
            l, r, n, f = c_idx - br - Wm // 2, c_idx + br + Wm // 2, c_idy - br - Hm // 2, c_idy + br + Hm // 2
            padding = padl, padr, padn, padf = (max(0, -l), max(0, r - W + 1), max(0, -n), max(0, f - H + 1))
            padding = 0, 0, *padding  # C channel should have no padding

            pts_glb = pts_glb_all[:, n + padn:f - padf + 1, l + padl:r - padr + 1, :]
            block = get_block(pts_glb, model, embedder, post_processor, Flags.chunk)
            # img_loss = 0
            optimizer.zero_grad()

            # deblur
            pred1 = torch.stack([forward_project(block, H_mtx,padding=padding) for H_mtx in H_mtxs[0]])
            target1 = images[0, :, c_idy - br:c_idy + br + 1, c_idx - br:c_idx + br + 1, :]

            # raw
            # pred2 = torch.stack([forward_project(block, H_mtx,padding=padding) for H_mtx in H_mtxs[1]])
            # target2 = images[1, :, c_idy - br:c_idy + br + 1, c_idx - br:c_idx + br + 1, :]

            loss1 = 1 - ssim_loss(pred1.squeeze(-1).unsqueeze(1), target1.squeeze(-1).unsqueeze(1))
            loss2 = img2mse(pred1.squeeze(-1).unsqueeze(1), target1.squeeze(-1).unsqueeze(1))

            loss = Flags.ssim_ratio * loss1 + loss2

            if Flags.L1_regu_weight!=0:
                L1_regu = block.abs().mean()
                loss = loss + Flags.L1_regu_weight * L1_regu
            if Flags.TV_regu_weight!=0:
                block_xx = block[:,:,2:] + block[:,:,:-2] - 2 * block[:,:,1:-1]
                block_yy = block[:,2:,:] + block[:,:-2,:] - 2 * block[:,1:-1,:]
                block_zz = block[2:,:,:] + block[:-2,:,:] - 2 * block[1:-1,:,:]
                block_xy = block[:-1,1:,1:] + block[:-1,:-1,:-1] - block[:-1,:-1,1:] - block[:-1,1:,:-1]
                block_xz = block[1:,:-1,1:] + block[:-1,:-1,:-1] - block[:-1,:-1,1:] - block[1:,:-1,:-1]
                block_yz = block[1:,1:,:-1] + block[:-1,:-1,:-1] - block[:-1,1:,:-1] - block[1:,:-1,:-1]
                TV_regu = block_xx.abs().mean() + block_yy.abs().mean() + block_zz.abs().mean() + 2 * (block_xy.abs().mean() + block_xz.abs().mean() + block_yz.abs().mean())
                loss = loss + Flags.TV_regu_weight * TV_regu

            # print('mean: %.4f' % torch.mean(block).cpu())
            loss.backward()
            optimizer.step()

            ###   update learning rate   ###
            decay_rate = 0.1
            decay_steps = Flags.lrate_decay * 1000
            new_lrate = Flags.lrate * (decay_rate ** ((step-1) / decay_steps))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate


            ###   intial leaky###
            if step == 500 and 'post_processor_old' in locals(): post_processor = post_processor_old

            ################################

            ## logging
            if step % Flags.i_print == 0:
                with torch.no_grad():
                    voxel = get_block(pts_glb_all, model, embedder, post_processor, chunk=Flags.chunk)

                    re_pred1 = torch.stack([forward_project(voxel, H_mtx) for H_mtx in H_mtxs[0]])

                    ssim_val_1 = ssim_loss(re_pred1.squeeze(-1).unsqueeze(1), images[0].squeeze(-1).unsqueeze(1))

                    re_prj2_loss = img2mse(re_pred1.squeeze(-1).unsqueeze(1), images[0].squeeze(-1).unsqueeze(1))

                    psnr_avg = mse2psnr(re_prj2_loss)

                    if psnr_avg > psnr or step % 5000 == 0:
                        if psnr_avg>psnr:
                            psnr = psnr_avg
                        tqdm_txt = f"[TRAIN] Iter: {step} Loss_fine: {re_prj2_loss.item()} PSNR: {psnr_avg.item()} SSIM: {ssim_val_1.item()}" \
                                   f"Intensity mean: {torch.mean(block).item()}"

                        tqdm.write(tqdm_txt)
                        # write into file
                        path = os.path.join(Flags.basedir, Flags.expname, 'logs.txt')
                        with open(path, 'a') as fp:
                            fp.write(tqdm_txt+'\n')
                        if Flags.shot_when_print:
                            if step % 5000 == 0:

                                tifffile.imwrite(os.path.join(Flags.basedir, Flags.expname, 'D1',f'ViewStack_%05d.tif'%step),
                                                 np.squeeze(re_pred1.cpu().numpy()))
                                tifffile.imwrite(os.path.join(Flags.basedir, Flags.expname, 'recon',f'Rendering_%05d.tif'%step),
                                             np.squeeze(voxel.cpu().numpy()[::-1, ...]))
                            else:
                                tifffile.imwrite(
                                    os.path.join(Flags.basedir, Flags.expname, 'D1', 'ViewStack_best.tif' ),
                                    np.squeeze(re_pred1.cpu().numpy()))
                                tifffile.imwrite(
                                    os.path.join(Flags.basedir, Flags.expname, 'recon', 'Rendering_best.tif'),
                                    np.squeeze(voxel.cpu().numpy()))
                                path = os.path.join(Flags.basedir, Flags.expname, 'INR_cp_best.tar')
                                save_dict = {
                                    'global_step': step - 1,
                                    'network_fn_state_dict': model.get_state(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                }
                                torch.save(save_dict, path)
                                print('Saved checkpoints at', path)


            writer.add_scalar(tag="loss/train_ssim_loss", scalar_value=loss1,
                              global_step=step)
            writer.add_scalar(tag="loss/train_mse_loss", scalar_value=loss2,
                              global_step=step)

            del loss1,loss2, loss, target1, pred1
        writer.close()
    else:
        ## Create model
        if Flags.weights_path is not None and Flags.weights_path != 'None':
            ckpts = [Flags.weights_path]
        else:
            ckpts = [os.path.join(Flags.basedir, Flags.expname, f) for f in
                     sorted(os.listdir(os.path.join(Flags.basedir, Flags.expname))) if 'AIR_cp_best' in f]

        print('Found ckpts', ckpts)
        if len(ckpts) > 0 and not Flags.no_reload:
            ckpt_path = ckpts[-1]
            print('Reloading from', ckpt_path)
        else:
            ckpt_path = None
            print("Creating new model...")

        model, net_ker,embedder, post_processor, optimizer, start = create_model_psf(Flags, (D, H, W), Flags.modeltype,
                                                                         Flags.embeddertype, Flags.postprocessortype,
                                                                         lr=Flags.lrate, weights_path=ckpt_path)
        kbase=torch.zeros(H_mtxs.shape[1],H_mtxs.shape[2], H_mtxs.shape[3],H_mtxs.shape[4],dtype=torch.complex64)
        for i in range(Nm):
            psf_view=H_mtxs[0][i]
            # tifffile.imwrite(f'output/psf/psf_raw_{i}.tif', psf_view.cpu().detach().numpy().astype(np.float32))
            # psf_view /= torch.sum(psf_view, dim=(1, 2), keepdim=True)
            psf_view=torch.fft.fftshift(psf_view)
            kbase[i,:,:,:]=torch.fft.fftn(psf_view, dim=(1, 2))
        psf = PsfGenerator(kbase,
                             units=(Flags.psf_dy, Flags.psf_dx),
                             na_detection=Flags.n_detection,
                             lam_detection=Flags.emission_wavelength,
                             n=Flags.n_obj)
        # post processor
        if (post_processor is torch.relu or post_processor is torch.relu_) and start < 500:
            post_processor_old, post_processor = post_processor, torch.nn.functional.leaky_relu
        start+=1
        psnr=0
        for step in trange(start, N_steps):

            # get random block center (c_idx,c_idy)
            c_idx = np.random.randint(br, W - br)
            c_idy = np.random.randint(br, H - br)

            # c_idx = images.shape[3] // 2
            # c_idy = images.shape[2] // 2

            # views_permute=np.random.permtation(N)
            l, r, n, f = c_idx - br - Wm // 2, c_idx + br + Wm // 2, c_idy - br - Hm // 2, c_idy + br + Hm // 2
            padding = padl, padr, padn, padf = (max(0, -l), max(0, r - W + 1), max(0, -n), max(0, f - H + 1))
            padding = 0, 0, *padding  # C channel should have no padding

            pts_glb = pts_glb_all[:, n + padn:f - padf + 1, l + padl:r - padr + 1, :]
            block = get_block(pts_glb, model, embedder, post_processor, Flags.chunk)
            wf = net_ker.k
            out_k_m = psf.aberration_psf(wf)
            optimizer.zero_grad()

            # abbreation
            target1 = images[0, :, c_idy - br:c_idy + br + 1, c_idx - br:c_idx + br + 1, :]
            pred1 = torch.stack([forward_project(block, H_mtx, padding=padding) for H_mtx in out_k_m], dim=0)
            loss1 = 1 - ssim_loss(pred1.squeeze(-1).unsqueeze(1), target1.squeeze(-1).unsqueeze(1))
            loss2 = img2mse(pred1.squeeze(-1).unsqueeze(1), target1.squeeze(-1).unsqueeze(1))
            loss = Flags.ssim_ratio * loss1+ loss2
            loss += single_mode_control(wf, 1, 0, 0) # quite crucial for suppressing unwanted defocus.

            if Flags.L1_regu_weight != 0:
                L1_regu = block.abs().mean()
                loss = loss + Flags.L1_regu_weight * L1_regu
            if Flags.TV_regu_weight != 0:
                block_xx = block[:, :, 2:] + block[:, :, :-2] - 2 * block[:, :, 1:-1]
                block_yy = block[:, 2:, :] + block[:, :-2, :] - 2 * block[:, 1:-1, :]
                block_zz = block[2:, :, :] + block[:-2, :, :] - 2 * block[1:-1, :, :]
                block_xy = block[:-1, 1:, 1:] + block[:-1, :-1, :-1] - block[:-1, :-1, 1:] - block[:-1, 1:, :-1]
                block_xz = block[1:, :-1, 1:] + block[:-1, :-1, :-1] - block[:-1, :-1, 1:] - block[1:, :-1, :-1]
                block_yz = block[1:, 1:, :-1] + block[:-1, :-1, :-1] - block[:-1, 1:, :-1] - block[1:, :-1, :-1]
                TV_regu = block_xx.abs().mean() + block_yy.abs().mean() + block_zz.abs().mean() + 2 * (
                            block_xy.abs().mean() + block_xz.abs().mean() + block_yz.abs().mean())
                loss = loss + Flags.TV_regu_weight * TV_regu

            loss.backward()
            optimizer.step()

            ###   update learning rate   ###
            new_lrate=[]
            decay_rate = 0.1
            decay_steps = Flags.lrate_decay * 1000
            new_lrate.append( Flags.lrate * (decay_rate ** ((step - 1) / decay_steps)))
            new_lrate.append( Flags.lrate_kenel * (decay_rate ** ((step - 1) / decay_steps)))
            for l,param_group in enumerate(optimizer.param_groups):
                param_group['lr'] = new_lrate[l]

            ###   intial leaky###
            if step == 500 and 'post_processor_old' in locals(): post_processor = post_processor_old

            if step % Flags.i_print == 0:
                with torch.no_grad():
                    voxel = get_block(pts_glb_all, model, embedder, post_processor, chunk=Flags.chunk)

                    re_pred1 = torch.stack([forward_project(voxel, H_mtx) for H_mtx in out_k_m])

                    ssim_val_1 = ssim_loss(re_pred1.squeeze(-1).unsqueeze(1), images[0].squeeze(-1).unsqueeze(1))

                    re_prj2_loss = img2mse(re_pred1.squeeze(-1).unsqueeze(1), images[0].squeeze(-1).unsqueeze(1))

                    psnr_avg = mse2psnr(re_prj2_loss)
                    if psnr_avg>psnr or step%5000==0:

                        psnr = psnr_avg

                        tqdm_txt = f"[TRAIN] Iter: {step} Loss_fine: {re_prj2_loss.item()} PSNR: {psnr_avg.item()} SSIM: {ssim_val_1.item()}" \
                                   f"Intensity mean: {torch.mean(block).item()}"

                        tqdm.write(tqdm_txt)
                        # write into file
                        path = os.path.join(Flags.basedir, Flags.expname, 'logs_aber.txt')
                        with open(path, 'a') as fp:
                            fp.write(tqdm_txt + '\n')
                        if Flags.shot_when_print:
                            # for l, param_group in enumerate(optimizer.param_groups):
                            #     print('learning rate for group %d: %.6f' % (l, param_group['lr']))
                            wf = net_ker.k
                            out_k_m = psf.aberration_psf(wf)
                            if step % 5000 == 0:
                                tifffile.imwrite(
                                    os.path.join(Flags.basedir, Flags.expname, 'D1', f'aberra_ViewStack_{step}.tif' ),
                                    np.squeeze(re_pred1.cpu().numpy()))

                                tifffile.imwrite(
                                    os.path.join(Flags.basedir, Flags.expname, 'recon', f'aberra_Rendering_step{step}.tif'),
                                    np.squeeze(voxel.cpu().numpy()))
                            else:
                                tifffile.imwrite(
                                    os.path.join(Flags.basedir, Flags.expname, 'D1', 'aberra_ViewStack_best.tif'),
                                    np.squeeze(re_pred1.cpu().numpy()))

                                tifffile.imwrite(
                                    os.path.join(Flags.basedir, Flags.expname, 'recon', 'aberra_Rendering_best.tif'),
                                    np.squeeze(voxel.cpu().numpy()))
                                for Vm, H_mtx in enumerate(out_k_m):
                                    tifffile.imwrite(
                                        os.path.join(Flags.basedir, Flags.expname, 'D2',
                                                     f'aberra_psf_%05d.tif' % Vm),
                                        H_mtx.cpu().numpy())
                                path = os.path.join(Flags.basedir, Flags.expname, 'AIR_cp_best.tar')
                                save_dict = {
                                    'global_step': step - 1,
                                    'network_fn_state_dict': model.get_state(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                }
                                torch.save(save_dict, path)
                                print('Saved checkpoints at', path)

            writer.add_scalar(tag="loss/train_ssim_loss", scalar_value=loss1,
                              global_step=step)

            del loss1, loss, target1, pred1
        visualize_3d_images(voxel.cpu().numpy(), Flags)