import torch
from model import PSDN
import PIL.Image as Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.nn import functional as F
import numpy as np
from io import BytesIO


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

        # Hook for gradients
        self.target_layer.register_forward_hook(self.forward_hook)
        self.target_layer.register_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activation = output

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, class_idx):
        # Global Average Pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        # Weighted sum of activations
        cam = torch.sum(weights * self.activation, dim=1).squeeze(0)
        cam = F.relu(cam)  # ReLU to ensure positive values
        # Normalize the heatmap
        cam -= cam.min()
        cam /= cam.max()
        return cam

# Load the model
model = PSDN(n_pixel_class=16)
model.load_state_dict(torch.load("psdn_model.pth"))
model.eval()

# Initialize Grad-CAM with the desired layer
target_layer = model.conv1
gradcam = GradCAM(model, target_layer)

# Inference with Grad-CAM
if __name__ == "__main__":
    source_path = r""
    source = Image.open(source_path).convert("L")

    buffer = BytesIO()
    source.save(buffer, "JPEG", quality=20)
    buffer.seek(0)
    source = Image.open(buffer)

    # 在h,w上用0 pad到32的倍数
    w, h = source.size
    new_w = w + 32 - w % 32
    new_h = h + 32 - h % 32


    transform = transforms.Compose([
        transforms.Pad((0, 0, new_w - w, new_h - h), fill=0),
        
        transforms.ToTensor()])
    source = transform(source).unsqueeze(0)

        # Forward pass
    output = model(source)
    predicted_class = torch.argmax(output, dim=1).item()

    # Backward pass to get gradients
    model.zero_grad()
    one_hot_output = torch.zeros_like(output)
    one_hot_output[0, predicted_class] = 1
    output.backward(gradient=one_hot_output)

    # Generate Grad-CAM heatmap
    heatmap = gradcam.generate_heatmap(predicted_class)

    # Visualize heatmap overlaid on the input image
    source_img = transforms.ToPILImage()(source.squeeze(0))
    heatmap = transforms.ToPILImage()(heatmap.unsqueeze(0))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title(f"Original Image, Predicted Pixel Size: {predicted_class+1}")
    plt.imshow(source_img, cmap="gray")

    plt.subplot(1, 2, 2)
    plt.title("Grad-CAM Heatmap")
    #plt.imshow(source_img, cmap="gray")
    plt.imshow(heatmap, cmap="jet", alpha=1)  # Overlay heatmap
    plt.show()
