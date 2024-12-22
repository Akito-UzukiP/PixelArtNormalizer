# Pixel Art Normalization Tool

This project is designed to detect the pixel size of pixel art images sourced from the internet and normalize their resolution through downsampling. The primary goal is to preprocess pixel art for subsequent training tasks. The model is specifically trained to handle distortions caused by JPG compression, offering robust predictions. However, note that downsampling will not magically remove JPG compression artifacts.

## Features

- **Pixel Size Detection**: Predict the pixel size of the input image using the trained CNN model.
- **Downsampling**: Perform NEAREST downsampling based on the predicted pixel size.
- **GIF Handling**: Automatically downsample all frames in a GIF and reassemble the output.
- **Batch Processing**: Process entire directories containing supported image formats (PNG, JPG, JPEG, GIF).

## Usage

1. **Install dependencies**:
   Ensure you have Python installed. Then install the required libraries:
   ```bash
   pip install torch torchvision pillow
   ```

2. **Run the tool**:
   ```bash
   python downsample.py input_path --model path_to_model.pth --output output_directory
   ```

   - `input_path`: Path to an image, GIF, or directory containing images.
   - `path_to_model.pth`: Path to the trained PSDN model file (default is `./psdn_model.pth`).
   - `output_directory`: Directory to save the processed images (default is `./output`).
   - `dest_pixel_size`: Size of the pixel block after resize (default is `1`).

3. **Examples**:
   - Process a single image:
     ```bash
     python main.py ./image.jpg --model ./psdn_model.pth --output ./output
     ```
   - Process a directory of images:
     ```bash
     python main.py ./images --model ./psdn_model.pth --output ./processed_images
     ```
   - Process a GIF:
     ```bash
     python main.py ./animation.gif --model ./psdn_model.pth --output ./output
     ```

---

# 像素艺术归一化工具

本项目旨在检测来自互联网的像素艺术图像的像素点大小，并通过下采样对其分辨率进行归一化。其主要目标是为后续训练任务预处理像素艺术图像。模型在训练时特别针对 JPG 压缩失真进行了优化，具有较强的鲁棒性。但需要注意的是，下采样不会神奇地消除 JPG 压缩失真。

## 功能

- **像素点大小检测**：使用训练好的 CNN 模型预测输入图像的像素点大小。
- **下采样**：根据预测的像素点大小进行 NEAREST 下采样。
- **GIF 处理**：自动下采样 GIF 的所有帧并重新组装。
- **批量处理**：支持处理目录下的所有支持格式的图像（PNG, JPG, JPEG, GIF）。

## 使用方法

1. **安装依赖**：
   确保已安装 Python，然后安装必要的库：
   ```bash
   pip install torch torchvision pillow
   ```

2. **运行工具**：
   ```bash
   python downsample.py input_path --model path_to_model.pth --output output_directory
   ```

   - `input_path`: 输入路径，可以是单个图像、GIF 或包含图像的目录。
   - `path_to_model.pth`: 训练好的 PSDN 模型文件路径（默认是 `./psdn_model.pth`）。
   - `output_directory`: 保存处理后图像的目录（默认是 `./output`）。
   - `dest_pixel_size`: 重采样后的像素点大小（默认是`1`）。

3. **示例**：
   - 处理单个图像：
     ```bash
     python main.py ./image.jpg --model ./psdn_model.pth --output ./output
     ```
   - 处理图像目录：
     ```bash
     python main.py ./images --model ./psdn_model.pth --output ./processed_images
     ```
   - 处理 GIF：
     ```bash
     python main.py ./animation.gif --model ./psdn_model.pth --output ./output
     ```

