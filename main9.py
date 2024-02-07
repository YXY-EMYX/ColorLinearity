#通过偏移到指定颜色色环直径实现图片色调的调换,偏移逻辑改为空间平面运动（即饱和度和色调）
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import colorsys
from scipy.optimize import minimize

def rgb_to_hsv(rgb_colors):
    return [colorsys.rgb_to_hsv(*[x / 255.0 for x in color]) for color in rgb_colors]

def hue_distance(h1, h2):
    d = abs(h1 - h2)
    return min(d, 1 - d)

def hs_distance(hs1, hs2):
    dh = hue_distance(hs1[0], hs2[0])
    ds = abs(hs1[1] - hs2[1])
    return np.sqrt(dh**2 + ds**2)

def find_optimal_diameter(hsv_colors):
    """ 寻找最佳直径使得所有颜色点到该直径的垂直距离总和最小 """
    def total_vertical_distance(angle):
        target_hue = angle / 360
        target_hue_opposite = (target_hue + 0.5) % 1

        total_distance = 0
        for h, s, _ in hsv_colors:
            distance_to_line = hue_distance(h, target_hue)
            distance_to_opposite_line = hue_distance(h, target_hue_opposite)
            vertical_distance = min(s * np.cos(distance_to_line * 2 * np.pi), s * np.cos(distance_to_opposite_line * 2 * np.pi))
            total_distance += vertical_distance

        return total_distance

    result = minimize(total_vertical_distance, x0=0, bounds=[(0, 360)])
    return result.x[0]

def adjust_hue_to_diameter(h, s, target_diameter):
    """ 将色调调整到最佳直径的垂直线上 """
    target_hue = target_diameter / 360
    target_hue_opposite = (target_hue + 0.5) % 1
    distance_to_line = hue_distance(h, target_hue)
    distance_to_opposite_line = hue_distance(h, target_hue_opposite)

    # 选择距离较近的直径，并计算新的饱和度
    if distance_to_line < distance_to_opposite_line:
        new_s = s * np.cos(distance_to_line * 2 * np.pi)
        return target_hue, new_s
    else:
        new_s = s * np.cos(distance_to_opposite_line * 2 * np.pi)
        return target_hue_opposite, new_s

def plot_color_wheel(hsv_colors, title):
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for h, s, _ in hsv_colors:
        rgb = colorsys.hsv_to_rgb(h, s, 1)
        rgb = [max(0, min(x, 1)) for x in rgb]
        ax.scatter(h * 2 * np.pi, s, color=[rgb], s=30)
    ax.set_title(title)
    plt.show()

def main():
    image_path = r"path_to_your_image.jpg"
    image = Image.open(image_path)
    img_array = np.array(image)
    hsv_image = np.array([colorsys.rgb_to_hsv(*[x / 255.0 for x in pixel]) for pixel in img_array.reshape((-1, 3))])

    kmeans = KMeans(n_clusters=30)
    img_array_reshaped = img_array.reshape((-1, 3))
    kmeans.fit(img_array_reshaped)
    dominant_colors = kmeans.cluster_centers_.astype(int)
    dominant_hsv_colors = rgb_to_hsv(dominant_colors)

    plot_color_wheel(dominant_hsv_colors, "Original Color Wheel")

    # 寻找最佳直径
    optimal_diameter = find_optimal_diameter(dominant_hsv_colors)

    # 调整聚类颜色至最佳直径
    adjusted_hsv_colors = [adjust_hue_to_diameter(h, s, optimal_diameter) + (v,) for h, s, v in dominant_hsv_colors]

    plot_color_wheel(adjusted_hsv_colors, "Adjusted Color Wheel")

    # 调整整个图像中每个像素的色调
    adjusted_hsv_image = np.array([adjust_hue_to_diameter(h, s, optimal_diameter) + (v,) for h, s, v in hsv_image])

    # 转换调整后的HSV颜色回RGB并重塑为原始图像的形状
    adjusted_rgb_image = np.array([colorsys.hsv_to_rgb(h, s, v) for h, s, v in adjusted_hsv_image])
    adjusted_rgb_image = (adjusted_rgb_image.reshape(img_array.shape) * 255).astype(np.uint8)

    # 保存和显示调整后的图像
    adjusted_image = Image.fromarray(adjusted_rgb_image)
    adjusted_image.save('adjusted_image.png')
    adjusted_image.show()

if __name__ == "__main__":
    main()
