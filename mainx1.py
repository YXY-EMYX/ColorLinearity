#简单比较图片质量，确保机器视角下图片质量没有明显下降
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def calculate_psnr(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_ssim(image1, image2):
    ssim_value = ssim(image1, image2, multichannel=True)
    return ssim_value

# 加载图像
image_path1 = r"path_to_your_image.jpg"  # 替换为第一张图片的路径
image_path2 = r"path_to_your_image.jpg" # 替换为第二张图片的路径

image1 = cv2.imread(image_path1)
image2 = cv2.imread(image_path2)

# 将图像转换为灰度（如果使用SSIM进行灰度图像比较）
image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# 计算PSNR和SSIM
psnr_value = calculate_psnr(image1, image2)
ssim_value = calculate_ssim(image1_gray, image2_gray)

print(f"PSNR: {psnr_value}")
print(f"SSIM: {ssim_value}")
