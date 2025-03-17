import numpy as np
import re
import os
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import tifffile
from skimage.transform import rescale
from skimage.exposure import adjust_gamma
def normalize(x,eps=1e-5):
    max_ = np.max(x)
    min_ =np.min(x)
    x = (x-min_) / (max_-min_+eps)
    return x

def load_imgs(path, regx='.*.tif', printable=False):
    file_list = os.listdir(path)
    return_list = []
    for _, f in enumerate(file_list):
        if re.search(regx, f):
            return_list.append(f)
    img_list = sorted(return_list)
    return img_list
def exists_or_mkdir(path, verbose=True):
    if not os.path.exists(path):
        os.makedirs(path)
        return False
    else:
        return True

def single_mode_control(wf, num, vmin, vmax):
    return torch.mean(F.relu(wf[:,num] - vmax) + F.relu(-wf[:,num] + vmin))


def tv_1d(img, axis='z', normalize=False):
    if axis == 'z':
        if not normalize:
            _variance = torch.sum(torch.abs(img[0:-1, :, :] - img[1::, :, :]))
        else:
            _variance = torch.mean(torch.abs(img[0:-1, :, :] - img[1::, :, :]))

    elif axis == 'y':
        if not normalize:
            _variance = torch.sum(torch.abs(img[:, 0:-1, :] - img[:, 1::, :]))
        else:
            _variance = torch.mean(torch.abs(img[:, 0:-1, :] - img[:, 1::, :]))

    else:
        if not normalize:
            _variance = torch.sum(torch.abs(img[:, :, 0:-1] - img[:, :, 1::]))
        else:
            _variance = torch.mean(torch.abs(img[:, :, 0:-1] - img[:, :, 1::]))

    return _variance

def write_tensor_to_file(tensor, file_path):
    with open(file_path, 'w') as f:
        for row in tensor:
            formatted_row = ' '.join(f"{value:.4f}" for value in row.tolist())
            f.write(formatted_row + '\n')
            # 将每一行的值转换为字符串并写入文件
            # f.write(' '.join(map(str, row.tolist())) + '\n')




def visualize_3d_images(img, Flags):
    # 尝试读取图像a和b
    try:
        a = tifffile.imread(os.path.join(Flags.datadir, r'view_stack\Deconv\deconv_iter15_DAO_1.tif'))
    except FileNotFoundError:
        a = None

    try:
        b = tifffile.imread(os.path.join(Flags.datadir, r'view_stack\Deconv\deconv_iter15_DAO_0.tif'))
    except FileNotFoundError:
        b = None

    # 定义一个函数来处理图像
    def process_image(image):
        if image is not None:
            # 首尾各删除4层
            image = image[3:-3]
            # 沿z轴做mip
            mip_z = np.max(image, axis=0)
            # 从上到下做reslice
            if 'exp' in Flags.datadir:
                reslice = image[:, 100:150]
            else:
                reslice = image
            # 沿y轴做mip
            mip_y = np.max(reslice, axis=1)
            mip_y_rescaled = rescale(mip_y.squeeze(), (float(Flags.psf_dz/Flags.psf_dx), 1), anti_aliasing=True)
            mip_z_corrected = adjust_gamma(np.clip(mip_z, 0, None), 0.7)
            mip_y_corrected = adjust_gamma(np.clip(mip_y_rescaled, 0, None), 0.8)
            return mip_z_corrected, mip_y_corrected
        return None, None

    # 处理图像
    img_mip_z, img_mip_y = process_image(img)
    a_mip_z, a_mip_y = process_image(a)
    b_mip_z, b_mip_y = process_image(b)

    # 创建图形
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 第一排展示沿z轴的mip
    axes[0, 0].imshow(img_mip_z, cmap='hot')
    axes[0, 0].set_title('AIR-net MIP xy')
    if a_mip_z is not None:
        axes[0, 1].imshow(a_mip_z, cmap='hot')
        axes[0, 1].set_title('DAO MIP xy')
    if b_mip_z is not None:
        axes[0, 2].imshow(b_mip_z, cmap='hot')
        axes[0, 2].set_title('Deconvolution MIP xy')

    # 第二排展示沿y轴的mip
    axes[1, 0].imshow(img_mip_y, cmap='hot')
    axes[1, 0].set_title('AIR-net MIP xz')
    if a_mip_y is not None:
        axes[1, 1].imshow(a_mip_y, cmap='hot')
        axes[1, 1].set_title('DAO MIP xz')
    if b_mip_y is not None:
        axes[1, 2].imshow(b_mip_y, cmap='hot')
        axes[1, 2].set_title('Deconvolution MIP xz')

    # 设置整体标题
    fig.suptitle('Visualization of 3D Images', fontsize=16)

    for ax in axes.flatten():
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
