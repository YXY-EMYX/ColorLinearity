#通过偏移到指定6块颜色半径色环实现图片色调的调换
from scipy.optimize import minimize
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import colorsys

# 转换RGB到HSV
def rgb_to_hsv(rgb_colors):
    hsv_colors = []
    for color in rgb_colors:
        normalized_rgb = tuple([x / 255.0 for x in color])
        hsv_colors.append(colorsys.rgb_to_hsv(*normalized_rgb))
    return hsv_colors

# 计算色调距离
def hue_distance(h1, h2):
    d = abs(h1 - h2)
    return min(d, 1 - d)

# 绘制色环
def plot_color_wheel(hsv_colors, title):
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for color in hsv_colors:
        ax.scatter(color[0] * 2 * np.pi, color[1], color=colorsys.hsv_to_rgb(*color), s=30)
    ax.set_title(title)
    plt.show()

# 加载图片
def load_image(image_path):
    image = Image.open(image_path)
    img_array = np.array(image)
    return img_array

# HSV颜色的聚类
def cluster_colors(img_array):
    hsv_image = np.array([colorsys.rgb_to_hsv(*[x / 255.0 for x in pixel]) for pixel in img_array.reshape((-1, 3))])
    kmeans = KMeans(n_clusters=120)
    kmeans.fit(img_array.reshape((-1, 3)))
    dominant_colors = kmeans.cluster_centers_.astype(int)
    hsv_colors = rgb_to_hsv(dominant_colors)
    return hsv_colors

# 计算给定区间的最佳半径
def calculate_optimal_radius(hsv_colors, section):
    def total_distance_to_radius(radius_angle):
        radius_hue = radius_angle / 360
        total_distance = 0
        for h, s, _ in hsv_colors:
            # 确保颜色处于正确的区间内
            if section[0] <= h < section[1] or (section[0] > section[1] and (h >= section[0] or h < section[1])):
                # 计算到半径的距离
                distance = hue_distance(h, radius_hue)
                total_distance += distance
        return total_distance

    # 优化以寻找最小的总距离
    result = minimize(total_distance_to_radius, x0=120 * (section[0] + section[1]), bounds=[(0, 360)])
    return result.x[0]

def adjust_hue_to_radius_within_section(h, s, target_radius, section):
    target_hue = target_radius / 360
    if (h >= section[0] and h < section[1]) or (section[0] > section[1] and (h >= section[0] or h < section[1])):
        return target_hue, s
    else:
        return h, s

def adjust_pixel_to_optimal_radius(h, s, v, optimal_radii, sections):
    # 判断像素属于哪个区域
    for i, section in enumerate(sections):
        if section[0] <= h < section[1] or (section[0] > section[1] and (h >= section[0] or h < section[1])):
            # 使用该区域的最佳半径进行调整
            adjusted_hue = optimal_radii[i] / 360
            return adjusted_hue, s, v
    return h, s, v  # 如果不在任何区域，保持原样

def main():
    image_path = r"path_to_your_image.jpg"
    img_array = load_image(image_path)
    hsv_colors = cluster_colors(img_array)

    # 将图像转换为HSV颜色空间
    hsv_image = np.array([colorsys.rgb_to_hsv(*[x / 255.0 for x in pixel]) for pixel in img_array.reshape((-1, 3))])

    # 绘制原始色环
    plot_color_wheel(hsv_colors, "Original Color Wheel")

    # 切割色环为六个部分
    sections = [(1 / 6, 1 / 3), (1 / 3, 1 / 2), (1 / 2, 2 / 3), (2 / 3, 5 / 6), (5 / 6, 1), (0, 1 / 6)]

    optimal_radii = []
    for section in sections:
        optimal_radius = calculate_optimal_radius(hsv_colors, section)
        optimal_radii.append(optimal_radius)

    print("Optimal Radii:", optimal_radii)
    # 使用计算出的最佳半径调整颜色
    adjusted_hsv_colors = []
    for section, radius in zip(sections, optimal_radii):
        if radius is not None:
            adjusted_section_colors = [adjust_hue_to_radius_within_section(h, s, radius, section) + (v,)
                                       for h, s, v in hsv_colors if (h >= section[0] and h < section[1]) or (
                                                   section[0] > section[1] and (h >= section[0] or h < section[1]))]
            adjusted_hsv_colors.extend(adjusted_section_colors)

    # 绘制调整后的色环
    plot_color_wheel(adjusted_hsv_colors, "Adjusted Color Wheel")

    # 使用计算出的最佳半径调整整个图像
    adjusted_hsv_image = np.array([
        adjust_pixel_to_optimal_radius(h, s, v, optimal_radii, sections)
        for h, s, v in hsv_image.reshape((-1, 3))
    ])

    # 转换调整后的HSV颜色回RGB并重塑为原始图像的形状
    adjusted_rgb_image = np.array([
        colorsys.hsv_to_rgb(h, s, v) for h, s, v in adjusted_hsv_image
    ])
    adjusted_rgb_image = (adjusted_rgb_image.reshape(img_array.shape) * 255).astype(np.uint8)

    # 保存和显示调整后的图像
    adjusted_image = Image.fromarray(adjusted_rgb_image)
    adjusted_image.save('adjusted_image.png')
    adjusted_image.show()

if __name__ == "__main__":
    main()

