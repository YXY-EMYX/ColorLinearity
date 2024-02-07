# ColorLinearity 彩色线性调整项目

## Preface 前言
These programs are based on an unproven hypothesis: images with color wheels that exhibit certain linearity on the color wheel have better visual effects compared to naturally distributed colors. The programs are designed to consider different scenarios for image color wheel distribution. Due to my limited mathematical foundation and understanding of colors, the methods and adjustments might not be entirely appropriate. In other words, the adjustments may not fully comply with the descriptions provided. Your understanding and tolerance are greatly appreciated.

这些程序基于一个未经证明的假设：图像色环颜色在色环上呈现一定的线性关系时视觉效果会相对于自然分布的颜色更好。程序考虑了不同情况下图片色环应该采用什么线性颜色方式分布。由于我数学基础和对颜色的了解有限，所以处理方法和调整措施可能并不十分妥当。换句话说，这些调整可能不完全符合提供的描述。请您谅解并包容。

## Suitable Scenarios 适合使用的场景
These programs are suitable for non-natural images or images with pure color structures. In such cases, the identification and modification of colors can make the image appear more natural and normal. In testing with natural and complex-colored images, the simplistic and crude treatment can result in images that are less natural and harmonious due to the complexity of human vision. Simply put, these programs can be used for processing some simple artistic works and correcting some color schemes.

这些程序适用于非自然图片或具有纯净颜色结构的图片（纯色环境）。在这些情况下，识别和修改颜色可以使图片看起来更自然正常。在自然和颜色复杂的图片的测试中，由于人类视觉的复杂性，这种简单粗暴的处理会使图片显得不太自然和谐。简单来说，可以用于处理一些简单的绘画作品和修正一些配色方案。

- Monochromatic Environment 纯色环境: Images without complex ambient light and color distribution.
- Natural Images 自然图片: Images directly sourced from natural environments.

纯色环境（生造词）：没有复杂环境光和颜色分布的图片。
自然图片（生造词）：直接在自然中取材形成的图片。

### Known Deficiencies 已知缺陷
- Forced adjustment of image colors might reduce the total number of colors in the image.
- May cause a slight decrease in image quality.
- May render images with complex color compositions unviewable.
- Might adjust multiple colors to the same color, creating color blocks. This issue will be mildly addressed in the upcoming v2 update, which may involve adjusting each color to a corresponding exclusive color with some differentiation or adding perturbations to increase color complexity.
- The processing results have certain unpredictability.
- Lack of clear standards for parameter adjustments.
- My limited knowledge in mathematics and colors might lead to inherent flaws in the program.
- Missing annotations (this might be addressed in v2).
- Some processing methods, such as saturation segmentation, have not been generalized to other linear distributions due to lack of modularization.

- 图片颜色的强制调整可能会减少图片中的总颜色数量。
- 可能导致图片质量略有下降。
- 可能使得具有复杂颜色构成的图片变得不可视。
- 可能将多种颜色调整为同一颜色，形成色块现象。这个问题将在即将到来的v2更新中得到轻微解决，可能的处理方法包括将每种颜色调整为对应的独有颜色，并进行一定的区分，或者添加扰动以增加颜色复杂度。
- 处理结果具有一定的不可预测性。
- 参数调整缺乏明确标准。
- 我在数学和颜色方面的知识有限，可能导致程序本身存在缺陷。
- 缺少注释（可能在v2版本中解决）。
- 由于缺乏模块化，一些处理方法，如饱和度分段，未在其他线性分布中普及。

## Description 描述
ColorLinearity is a Python project designed for color analysis and adjustment in images. Its primary function is to rearrange the colors within an image so that they align linearly on a color wheel, creating a visually harmonious effect.

彩色线性调整项目是一个用于图像中颜色分析和调整的Python项目。其主要功能是在图像中重新排列颜色，使它们在色轮上线性对齐，从而创造出视觉上的和谐效果。

## Features 功能
- Analyzes the dominant colors in an image.
- Adjusts the colors to align them linearly on the color wheel.
- Provides visualizations of the color adjustments.

- 分析图像中的主要颜色。
- 调整颜色以使其在色轮上线性对齐。
- 提供颜色调整的可视化展示。

## Installation 安装方法
To run this project, you need to have Python installed on your system. Clone the repository and install the required dependencies:

要运行这个项目，你需要在你的系统上安装Python。克隆仓库并安装所需的依赖项：
```bash
git clone https://github.com/YXY-EMYX/ColorLinearity
```
```bash
cd ColorLinearity
```
```bash
pip install -r requirements.txt
```

## Usage 使用方法
Run the script with the desired image file. Replace `path_to_your_image.jpg` in the script with the path to your image file.

使用所需的图像文件运行脚本。请将脚本中的 `path_to_your_image.jpg` 替换为您的图像文件路径。

## Requirements 系统要求
- Python 3.x
- Matplotlib
- NumPy
- Scikit-learn
- Pillow

For detailed versions, see `requirements.txt`.

- Python 3.x
- Matplotlib
- NumPy
- Scikit-learn
- Pillow

具体版本详见 `requirements.txt`。

## License 许可证
[MIT](https://choosealicense.com/licenses/mit/)

