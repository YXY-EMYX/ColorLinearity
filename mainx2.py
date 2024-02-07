#比较图片质量2
import cv2
import numpy as np

def calculate_image_sharpness(image):
    # 将图像转换为灰度
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 使用拉普拉斯算子
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    # 计算方差
    variance = laplacian.var()
    return variance

# 加载图像
image_path1 = r"path_to_your_image.jpg"  # 替换为第一张图片的路径
image_path2 = r"path_to_your_image.jpg" # 替换为第二张图片的路径

image1 = cv2.imread(image_path1)
image2 = cv2.imread(image_path2)

# 计算清晰度
sharpness1 = calculate_image_sharpness(image1)
sharpness2 = calculate_image_sharpness(image2)

print(f"Image 1 Sharpness: {sharpness1}")
print(f"Image 2 Sharpness: {sharpness2}")
