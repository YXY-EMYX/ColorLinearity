#通过偏移到色段的颜色色环实现图片色调的调换，采用了极坐标变换，强制移动到线上
import numpy as np
import colorsys
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def rgb_to_hsv(rgb_colors):
    return [colorsys.rgb_to_hsv(*[x / 255.0 for x in color]) for color in rgb_colors]

def find_closest_point_on_line(hsv_color, line_points):
    # 色环上点的坐标（在极坐标系中）
    h, s = hsv_color[0], hsv_color[1]
    x, y = s * np.cos(h * 2 * np.pi), s * np.sin(h * 2 * np.pi)

    # 直线上的两点坐标
    x1, y1 = line_points[0]
    x2, y2 = line_points[1]

    # 计算直线方程系数
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2

    # 计算点到直线的最近点
    closest_x = (B * (B * x - A * y) - A * C) / (A * A + B * B)
    closest_y = (A * (-B * x + A * y) - B * C) / (A * A + B * B)

    # 将坐标转换回色调和饱和度
    closest_angle = np.arctan2(closest_y, closest_x) / (2 * np.pi) % 1
    closest_saturation = np.sqrt(closest_x**2 + closest_y**2)

    # 如果最近点在色环边界之外，将饱和度限制在1以内
    if closest_saturation > 1:
        closest_saturation = 1

    return closest_angle, closest_saturation, hsv_color[2]

# 绘制色环及直线
def plot_color_wheel_with_line(hsv_colors, line_hues, title):
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # 绘制色环
    for color in hsv_colors:
        ax.scatter(color[0] * 2 * np.pi, color[1], color=colorsys.hsv_to_rgb(*color), s=30)

    # 计算直线两端点的坐标
    x1, y1 = np.cos(line_hues[0] * 2 * np.pi), np.sin(line_hues[0] * 2 * np.pi)
    x2, y2 = np.cos(line_hues[1] * 2 * np.pi), np.sin(line_hues[1] * 2 * np.pi)

    # 绘制直线
    ax.plot([line_hues[0] * 2 * np.pi, line_hues[1] * 2 * np.pi], [1, 1], color='black')

    ax.set_title(title)
    plt.show()

def main():
    image_path = r"path_to_your_image.jpg"
    image = Image.open(image_path)
    img_array = np.array(image)

    kmeans = KMeans(n_clusters=30)
    img_array_reshaped = img_array.reshape((-1, 3))
    kmeans.fit(img_array_reshaped)
    dominant_colors = kmeans.cluster_centers_.astype(int)

    hsv_colors = rgb_to_hsv(dominant_colors)

    # 设置直线的两个端点的色调值
    hue1, hue2 = 0.1, 0.9  # 这里可以根据需要自行调整
    line_hues = [hue1, hue2]

    plot_color_wheel_with_line(hsv_colors, line_hues, "Original Color Wheel")
    # 这里可以根据需要自行调整
    line_points = [(np.cos(hue1 * 2 * np.pi), np.sin(hue1 * 2 * np.pi)),
                   (np.cos(hue2 * 2 * np.pi), np.sin(hue2 * 2 * np.pi))]

    adjusted_hsv_colors = [find_closest_point_on_line(color, line_points) for color in hsv_colors]
    plot_color_wheel_with_line(adjusted_hsv_colors, line_hues,"Adjusted Color Wheel")

    hsv_image = np.array([colorsys.rgb_to_hsv(*(x / 255.0 for x in pixel)) for pixel in img_array.reshape((-1, 3))])
    adjusted_hsv_image = np.array([find_closest_point_on_line(color, line_points) for color in hsv_image])

    adjusted_rgb_image = np.array([colorsys.hsv_to_rgb(*pixel) for pixel in adjusted_hsv_image])
    adjusted_rgb_image = (adjusted_rgb_image.reshape(img_array.shape) * 255).astype(np.uint8)

    adjusted_image = Image.fromarray(adjusted_rgb_image)
    adjusted_image.save('adjusted_image.png')
    adjusted_image.show()

if __name__ == "__main__":
    main()
