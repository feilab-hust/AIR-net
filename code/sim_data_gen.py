
from psf_gen_abre import PsfGenerator3D
import mat73
from core.utils.utils import *
##please run on cpu,which define the device to cpu in glbSettings.py!!!!!
features_path=r'data/sim_data/HS.mat'
volume_path=r'twophoton_data/GT.tif'

from core.utils.LF import *
view_num=9
psf_dz=0.2
psf_dy=1.8
psf_dx=1.8
n_detection=0.13
n_obj=1.33
max_val=0.4
emission_wavelength=1

voxel=tifffile.imread(volume_path)
psf_save_path=r'twophoton_data/psf_GT'
LF_save_path=r'twophoton_data/sim_LF'
os.makedirs(psf_save_path,exist_ok=True)
os.makedirs(LF_save_path,exist_ok=True)
if '.mat' in features_path:
    mat = mat73.loadmat(features_path)
    for key in mat:
        H = torch.from_numpy(mat[key])  # (H,W,D)
        break
else:
    H=torch.from_numpy(np.load(features_path))
kbase = torch.zeros(H.shape[0], H.shape[1], H.shape[2], H.shape[3], dtype=torch.complex64)
for i in range(view_num):
    psf_view = H[i]
    # tifffile.imwrite(f'output/psf/psf_raw_{i}.tif', psf_view.cpu().detach().numpy().astype(np.float32))
    psf_view = torch.fft.fftshift(psf_view)
    kbase[i, :, :, :] = torch.fft.fftn(psf_view,dim=(1, 2))
psf = PsfGenerator3D(kbase,
                     units=(psf_dz, psf_dy, psf_dx),
                     na_detection=n_detection,
                     lam_detection=emission_wavelength,
                     n=n_obj)
k4=torch.rand(view_num,1)
k5=torch.full((view_num,1), fill_value=0.5)
k6_15=torch.rand((view_num,10))
wf=torch.cat((k4,k5,k6_15),1)
wf=2*max_val*wf-max_val
voxel=torch.from_numpy(voxel[:,:,:,np.newaxis].astype(np.float32))
out_k_m = psf.coherent_psf(wf)
re_pred1 = torch.stack([forward_project(voxel, H_mtx) for H_mtx in out_k_m])
view_stack_aber=re_pred1.detach().cpu().numpy().astype(np.float32)
tifffile.imwrite(LF_save_path+'/view_stack_aber.tif', view_stack_aber)
H_tensor=H.to(torch.float32)
re_pred2 = torch.stack([forward_project(voxel, H_mtx) for H_mtx in H_tensor])
view_stack_raw=re_pred1.detach().cpu().numpy().astype(np.float32)
tifffile.imwrite(LF_save_path+'/view_stack_raw.tif', re_pred2.detach().cpu().numpy().astype(np.float32))
for i in range(out_k_m.shape[0]):
    tifffile.imwrite(psf_save_path+f'/psf_aber{i}.tif', out_k_m[i].detach().cpu().numpy().astype(np.float32))
