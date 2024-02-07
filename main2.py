#实现色环色调的调整
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import colorsys
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar

# Load the image
image_path = r"path_to_your_image.jpg"  # Replace with your image path
image = Image.open(image_path)

# Resize the image for faster processing
max_dimension = max(image.size)
scaling_factor = max_dimension / 300
new_size = (int(image.size[0] / scaling_factor), int(image.size[1] / scaling_factor))
image_small = image.resize(new_size, Image.Resampling.LANCZOS)

# Convert resized image to numpy array and reshape it to a 2d array for KMeans
img_array_small = np.array(image_small)
img_array_small = img_array_small.reshape((-1, 3))

# Use KMeans to find the most dominant color clusters on the resized image
kmeans = KMeans(n_clusters=6)
kmeans.fit(img_array_small)

# Get the cluster centers (dominant colors)
dominant_colors = kmeans.cluster_centers_.astype(int)


# Convert the dominant RGB colors to HSV
def rgb_to_hsv(rgb_colors):
    hsv_colors = []
    for color in rgb_colors:
        # Normalize the RGB values to range 0-1
        normalized_rgb = tuple([x / 255.0 for x in color])
        # Convert to HSV
        hsv_colors.append(colorsys.rgb_to_hsv(*normalized_rgb))
    return hsv_colors


hsv_colors = rgb_to_hsv(dominant_colors)


# Function to plot a solid color wheel with points
def plot_color_wheel_with_points(hsv_colors, title='Color Wheel with Points'):
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Create the radial and angular coordinates for the color wheel
    radius = np.linspace(0, 1, 100)
    angle = np.linspace(0, 2 * np.pi, 360)
    radius, angle = np.meshgrid(radius, angle)

    # Flatten the arrays for plotting
    radius = radius.flatten()
    angle = angle.flatten()

    # Convert to RGB for plotting
    hsv_tuples = np.stack((angle / (2 * np.pi), radius, np.ones_like(angle)), axis=1)
    rgb_colors = np.array([colorsys.hsv_to_rgb(*hsv) for hsv in hsv_tuples])

    # Plot the color wheel as scatter plot
    ax.scatter(angle, radius, c=rgb_colors, s=2, alpha=1)

    # Plot the provided color points
    for color in hsv_colors:
        hue_angle = color[0] * 2 * np.pi
        saturation_radius = color[1]
        ax.scatter(hue_angle, saturation_radius, color=colorsys.hsv_to_rgb(*color), s=100, edgecolors='white')

    # Remove grid, ticks, and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.spines['polar'].set_visible(False)

    # Set the title
    plt.title(title, y=1.08)
    plt.show()


# Plot the color wheel with the points from the image
plot_color_wheel_with_points(hsv_colors)

from scipy.optimize import minimize_scalar

def hue_to_point(hue):
    # 色调值转换为单位圆上的二维点
    angle = hue * 2 * np.pi
    return np.cos(angle), np.sin(angle)

def calculate_total_distance(hsv_colors, center_line_angle):
    total_distance = 0
    for h, s, v in hsv_colors:
        point = hue_to_point(h)
        # 直线的单位向量
        line_vector = hue_to_point(center_line_angle / 360)
        # 计算点到直线的距离
        distance = np.cross(line_vector, point)
        total_distance += abs(distance)
    return total_distance

# 优化过程寻找最佳角度
result = minimize_scalar(lambda x: calculate_total_distance(hsv_colors, x), bounds=(0, 360), method='bounded')
best_angle = result.x

# 根据最佳角度调整颜色
adjusted_hsv_colors = []
for h, s, v in hsv_colors:
    original_angle = h * 360
    adjusted_angle = best_angle
    if abs(original_angle - adjusted_angle) > 90 and abs(original_angle - adjusted_angle) < 270:
        # 如果原始角度与最佳角度的差大于90度且小于270度，则选择对径点
        adjusted_angle = (best_angle + 180) % 360
    # 转换回色调值
    adjusted_hue = adjusted_angle / 360
    adjusted_hsv_colors.append((adjusted_hue, s, v))

# 绘制原始色环和调整后的色环
plot_color_wheel_with_points(hsv_colors, 'Original Color Wheel with Points')
plot_color_wheel_with_points(adjusted_hsv_colors, 'Adjusted Color Wheel with Points')
