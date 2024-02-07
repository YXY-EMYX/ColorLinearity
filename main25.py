#通过偏移到色段的颜色色环实现图片色调的调换，采用了极坐标变换，线为圆上任意移动的线但所有颜色到线的距离最短
import numpy as np
import colorsys
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.optimize import minimize

def rgb_to_hsv(rgb_colors):
    return [colorsys.rgb_to_hsv(*[x / 255.0 for x in color]) for color in rgb_colors]

def hue_distance(h1, h2):
    d = abs(h1 - h2)
    return min(d, 1 - d)

def plot_color_wheel(hsv_colors, title):
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for color in hsv_colors:
        ax.scatter(color[0] * 2 * np.pi, color[1], color=colorsys.hsv_to_rgb(*color), s=30)
    ax.set_title(title)
    plt.show()

def load_image(image_path):
    image = Image.open(image_path)
    img_array = np.array(image)
    return img_array

def cluster_colors(img_array):
    hsv_image = np.array([colorsys.rgb_to_hsv(*[x / 255.0 for x in pixel]) for pixel in img_array.reshape((-1, 3))])
    kmeans = KMeans(n_clusters=30)
    kmeans.fit(img_array.reshape((-1, 3)))
    dominant_colors = kmeans.cluster_centers_.astype(int)
    return rgb_to_hsv(dominant_colors)

def calculate_total_distance_to_line(line_points, hsv_colors):
    total_distance = 0
    for h, s, _ in hsv_colors:
        x, y = s * np.cos(h * 2 * np.pi), s * np.sin(h * 2 * np.pi)
        x1, y1 = line_points[0]
        x2, y2 = line_points[1]
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2
        denominator = np.sqrt(A**2 + B**2)

        if denominator != 0:
            distance = abs(A * x + B * y + C) / denominator
            total_distance += distance

    return total_distance

def find_optimal_line(hsv_colors):
    init_line_points = [(1, 0), (-1, 0)]

    def objective(line_params):
        hue1, hue2 = line_params
        line_points = [(np.cos(hue1 * 2 * np.pi), np.sin(hue1 * 2 * np.pi)),
                       (np.cos(hue2 * 2 * np.pi), np.sin(hue2 * 2 * np.pi))]
        return calculate_total_distance_to_line(line_points, hsv_colors)

    result = minimize(objective, x0=[0.1, 0.9], bounds=[(0, 1), (0, 1)])
    optimal_hues = result.x
    return [(np.cos(optimal_hues[0] * 2 * np.pi), np.sin(optimal_hues[0] * 2 * np.pi)),
            (np.cos(optimal_hues[1] * 2 * np.pi), np.sin(optimal_hues[1] * 2 * np.pi))]

def adjust_color_to_line(hsv_color, line_points):
    x, y = hsv_color[1] * np.cos(hsv_color[0] * 2 * np.pi), hsv_color[1] * np.sin(hsv_color[0] * 2 * np.pi)
    x1, y1 = line_points[0]
    x2, y2 = line_points[1]
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    denominator = A * A + B * B

    if denominator > 1e-6:
        closest_x = (B * (B * x - A * y) - A * C) / denominator
        closest_y = (A * (-B * x + A * y) - B * C) / denominator
        closest_angle = np.arctan2(closest_y, closest_x) / (2 * np.pi) % 1
        return closest_angle, hsv_color[1], hsv_color[2]
    else:
        return hsv_color

def main():
    image_path = r"path_to_your_image.jpg"
    img_array = load_image(image_path)
    hsv_colors = cluster_colors(img_array)
    plot_color_wheel(hsv_colors, "Original Color Wheel")

    optimal_line_points = find_optimal_line(hsv_colors)

    adjusted_hsv_colors = [adjust_color_to_line(color, optimal_line_points) for color in hsv_colors]
    plot_color_wheel(adjusted_hsv_colors, "Adjusted Color Wheel")

    hsv_image = np.array([colorsys.rgb_to_hsv(*[x / 255.0 for x in pixel]) for pixel in img_array.reshape((-1, 3))])
    adjusted_hsv_image = np.array([adjust_color_to_line(color, optimal_line_points) for color in hsv_image])

    adjusted_rgb_image = np.array([colorsys.hsv_to_rgb(*pixel) for pixel in adjusted_hsv_image])
    adjusted_rgb_image = (adjusted_rgb_image.reshape(img_array.shape) * 255).astype(np.uint8)

    adjusted_image = Image.fromarray(adjusted_rgb_image)
    adjusted_image.save('adjusted_image.png')
    adjusted_image.show()

if __name__ == "__main__":
    main()
