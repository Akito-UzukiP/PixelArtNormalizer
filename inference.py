import torch
from model import PSDN
import PIL.Image as Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
# Load the model
model = PSDN(n_pixel_class=16)
model.load_state_dict(torch.load("psdn_model.pth"))
model.eval()

# Inference
if __name__ == "__main__":
    source_path = r"test.jpg"
    # if GIF, load the first frame
    source = Image.open(source_path).convert("L")
    transform = transforms.Compose([transforms.ToTensor()])
    source = transform(source).unsqueeze(0)

    with torch.no_grad():
        result = model.predict(source)
        pixel_size = result.item() + 1
        # Display the image
        plt.imshow(transforms.ToPILImage()(source.squeeze(0)))
        plt.title(f"Predicted Pixel Size: {pixel_size}")
        plt.show()
