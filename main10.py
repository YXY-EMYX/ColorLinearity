#通过偏移到指定颜色色环直径实现图片色调的调换，添加了饱和度权重，饱和度较高的颜色对最佳直径的确定有更大的影响。这意味着在调整图像色调时，这些饱和度高的颜色相比于其他颜色变化更小。
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import colorsys
from scipy.optimize import minimize

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

# 计算总距离并考虑饱和度的影响
def total_distance_weighted_by_saturation(hsv_colors, line_angle):
    total_distance = 0
    line_hue = line_angle / 360
    for h, s, v in hsv_colors:
        distance_to_line = hue_distance(h, line_hue)
        distance_to_opposite_line = hue_distance(h, (line_hue + 0.5) % 1)
        weighted_distance = min(distance_to_line, distance_to_opposite_line) * s  # 使用饱和度作为权重
        total_distance += weighted_distance
    return total_distance


# 调整色调至直径上
def adjust_hue_to_diameter(hue, target_hue):
    if hue_distance(hue, target_hue) < hue_distance(hue, (target_hue + 0.5) % 1):
        return target_hue
    else:
        return (target_hue + 0.5) % 1

# 绘制色环
def plot_color_wheel(hsv_colors, title):
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for color in hsv_colors:
        ax.scatter(color[0] * 2 * np.pi, color[1], color=colorsys.hsv_to_rgb(*color), s=30)
    ax.set_title(title)
    plt.show()

# 加载图片
image_path = r"path_to_your_image.jpg"
image = Image.open(image_path)
img_array = np.array(image)

# 聚类
kmeans = KMeans(n_clusters=100)  # 减少聚类数
img_array_reshaped = img_array.reshape((-1, 3))
kmeans.fit(img_array_reshaped)
dominant_colors = kmeans.cluster_centers_.astype(int)

# 转换为HSV
hsv_colors = rgb_to_hsv(dominant_colors)

# 绘制调整前的色环
plot_color_wheel(hsv_colors, "Original Color Wheel")

# 使用优化算法找到使总距离最小的直径角度
result = minimize(lambda x: total_distance_weighted_by_saturation(hsv_colors, x), x0=0, bounds=[(0, 360)])
optimal_line_angle = result.x[0]

# 调整色调至直径上
adjusted_hsv_colors = [(adjust_hue_to_diameter(h, optimal_line_angle / 360), s, v) for h, s, v in hsv_colors]

# 绘制调整后的色环
plot_color_wheel(adjusted_hsv_colors, "Adjusted Color Wheel")

# 调整整个图像的颜色
hsv_image = np.array([colorsys.rgb_to_hsv(pixel[0]/255., pixel[1]/255., pixel[2]/255.) for pixel in img_array.reshape((-1, 3))])
adjusted_hsv_image = np.array([(
    adjust_hue_to_diameter(h, optimal_line_angle / 360),
    s,
    v
) for h, s, v in hsv_image])

# 转换回RGB并重塑为原始图像形状
adjusted_rgb_image = np.array([colorsys.hsv_to_rgb(*pixel) for pixel in adjusted_hsv_image])
adjusted_rgb_image = (adjusted_rgb_image.reshape(img_array.shape) * 255).astype(np.uint8)

# 保存和显示调整后的图像
adjusted_image = Image.fromarray(adjusted_rgb_image)
adjusted_image.save('adjusted_image.png')
adjusted_image.show()
