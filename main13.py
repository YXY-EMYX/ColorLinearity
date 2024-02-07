#通过偏移到指定颜色色环半径实现图片色调的调换，算法做了简化，是粗估
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import colorsys
import numpy as np
import itertools
import random

# 转换RGB到HSV
def rgb_to_hsv(rgb_colors):
    return [colorsys.rgb_to_hsv(*[x / 255.0 for x in color]) for color in rgb_colors]

# 计算色调距离
def hue_distance(h1, h2):
    d = abs(h1 - h2)
    return min(d, 1 - d)

def find_main_hues(hsv_image, num_clusters=30):
    img_array_reshaped = hsv_image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(img_array_reshaped)
    return kmeans.cluster_centers_

def calculate_optimal_radii(main_hues, num_radii=2, angle_step=10, perturbation=5):
    radii_angles = [angle + random.uniform(-perturbation, perturbation) for angle in np.arange(0, 360, angle_step)]
    best_radii = []
    min_distance = float('inf')
    for combination in itertools.combinations(radii_angles, num_radii):
        total_dist = sum(min(hue_distance(hue[0], r / 360) for r in combination) for hue in main_hues)
        if total_dist < min_distance:
            min_distance = total_dist
            best_radii = combination
    return best_radii

def adjust_hue_to_closest_radii(h, s, v, radii):
    # 计算与每个半径的色调距离
    distances = [hue_distance(h, r / 360) for r in radii]
    # 找到最近的半径
    closest_radii_index = np.argmin(distances)
    # 调整色调至最近的半径
    adjusted_hue = radii[closest_radii_index] / 360
    return adjusted_hue, s, v


# 绘制色环
def plot_color_wheel(hsv_colors, title):
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for color in hsv_colors:
        ax.scatter(color[0] * 2 * np.pi, color[1], color=colorsys.hsv_to_rgb(*color), s=30)
    ax.set_title(title)
    plt.show()


def main():
    image_path = r"path_to_your_image.jpg"
    image = Image.open(image_path)
    img_array = np.array(image)

    # 转换整个图像到HSV颜色空间
    hsv_image = np.array([colorsys.rgb_to_hsv(*[x / 255.0 for x in pixel]) for pixel in img_array.reshape((-1, 3))])

    # 聚类以获取主导颜色
    kmeans = KMeans(n_clusters=30)
    img_array_reshaped = img_array.reshape((-1, 3))
    kmeans.fit(img_array_reshaped)
    dominant_colors = kmeans.cluster_centers_.astype(int)

    # 将主导颜色转换为HSV并绘制原始色环
    dominant_hsv_colors = rgb_to_hsv(dominant_colors)
    plot_color_wheel(dominant_hsv_colors, "Original Color Wheel")

    # 找到最佳半径
    optimal_radii = calculate_optimal_radii(dominant_hsv_colors)

    # 调整聚类颜色至最近的最佳半径并绘制调整后的色环
    adjusted_hsv_colors = [adjust_hue_to_closest_radii(h, s, v, optimal_radii) for h, s, v in dominant_hsv_colors]
    plot_color_wheel(adjusted_hsv_colors, "Adjusted Color Wheel")

    # 调整每个像素的颜色至最近的最佳半径
    adjusted_hsv_image = np.array([adjust_hue_to_closest_radii(h, s, v, optimal_radii) for h, s, v in hsv_image])
    adjusted_rgb_image = np.array([colorsys.hsv_to_rgb(*pixel) for pixel in adjusted_hsv_image])
    adjusted_rgb_image = (adjusted_rgb_image.reshape(img_array.shape) * 255).astype(np.uint8)
    adjusted_image = Image.fromarray(adjusted_rgb_image)
    adjusted_image.save('adjusted_image.png')
    adjusted_image.show()

if __name__ == "__main__":
    main()
