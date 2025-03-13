import numpy as np
import matplotlib.pyplot as plt
import tifffile

def visualize_3d_images(img, Flags):
    # 尝试读取图像a和b
    try:
        a = tifffile.imread(f"{directory}/a.tif")
    except FileNotFoundError:
        a = None

    try:
        b = tifffile.imread(f"{directory}/b.tif")
    except FileNotFoundError:
        b = None

    # 定义一个函数来处理图像
    def process_image(image):
        if image is not None:
            # 首尾各删除4层
            image = image[4:-4]
            # 沿z轴做mip
            mip_z = np.max(image, axis=0)
            # 从上到下做reslice
            reslice = image[:, 50:80]
            # 沿y轴做mip
            mip_y = np.max(reslice, axis=1)
            return mip_z, mip_y
        return None, None

    # 处理图像
    img_mip_z, img_mip_y = process_image(img)
    a_mip_z, a_mip_y = process_image(a)
    b_mip_z, b_mip_y = process_image(b)

    # 创建图形
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 第一排展示沿z轴的mip
    axes[0, 0].imshow(img_mip_z, cmap='hot')
    axes[0, 0].set_title('Image MIP Z')
    if a_mip_z is not None:
        axes[0, 1].imshow(a_mip_z, cmap='hot')
        axes[0, 1].set_title('A MIP Z')
    if b_mip_z is not None:
        axes[0, 2].imshow(b_mip_z, cmap='hot')
        axes[0, 2].set_title('B MIP Z')

    # 第二排展示沿y轴的mip
    axes[1, 0].imshow(img_mip_y, cmap='hot')
    axes[1, 0].set_title('Image MIP Y')
    if a_mip_y is not None:
        axes[1, 1].imshow(a_mip_y, cmap='hot')
        axes[1, 1].set_title('A MIP Y')
    if b_mip_y is not None:
        axes[1, 2].imshow(b_mip_y, cmap='hot')
        axes[1, 2].set_title('B MIP Y')

    # 隐藏没有图像的子图
    for ax in axes.flatten():
        ax.axis('off')

    plt.tight_layout()
    plt.show()

