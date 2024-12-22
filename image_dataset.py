import os
import random
from PIL import Image, ImageFilter
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset
from io import BytesIO

class OptimizedImageProcessingDataset(Dataset):
    def __init__(self, root_dir, upscale_factor=2, crop_prob=0.5, interpolation_prob=0.5, jpeg_quality=85, jpeg_prob=0.05):
        """
        Args:
            root_dir (str): Directory with all the raw PNG images.
            upscale_factor (int): Maximum upscale factor for resizing images.
            crop_prob (float): Probability of performing cropping.
            interpolation_prob (float): Probability of using interpolation.
            jpeg_quality (int): JPEG compression quality (1-100).
        """
        self.root_dir = root_dir
        self.upscale_factor = upscale_factor
        self.crop_prob = crop_prob
        self.jpeg_prob = jpeg_prob
        self.interpolation_prob = interpolation_prob
        self.jpeg_quality = jpeg_quality
        self.interpolation_map = {
            "NEAREST": Image.NEAREST,
            "BILINEAR": Image.BILINEAR
        }
        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.png')]

        self.preprocess = transforms.Compose([
            transforms.Pad((0, 0, 512, 512), fill=0),  
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        # 灰度图
        img = Image.open(img_path).convert("L")

        # Perform random cropping
        if random.random() < self.crop_prob:
            width, height = img.size
            crop_scale = random.uniform(0.2, 1.0)
            crop_width = int(width * crop_scale)
            crop_height = int(height * crop_scale)
            left = random.randint(0, width - crop_width)
            top = random.randint(0, height - crop_height)
            img = img.crop((left, top, left + crop_width, top + crop_height))

        # Random upscale
        actual_upscale_factor = random.randint(1, self.upscale_factor)
        new_size = (img.width * actual_upscale_factor, img.height * actual_upscale_factor)
        interpolation_method = (
            self.interpolation_map["NEAREST"]
            if random.random() < self.interpolation_prob
            else self.interpolation_map["BILINEAR"]
        )
        img = img.resize(new_size, interpolation_method)

        # crop to 512x512
        img = transforms.CenterCrop(512)(img)

        # Add JPEG compression distortion in memory
        if random.random() < self.jpeg_prob:
            buffer = BytesIO()
            img.save(buffer, "JPEG", quality=self.jpeg_quality)
            buffer.seek(0)
            img = Image.open(buffer)

        # Final processing: crop/pad and convert to tensor
        img_tensor = self.preprocess(img)

        # Metadata
        metadata = {
            "actual_upscale_factor": actual_upscale_factor,
            "crop_prob": self.crop_prob,
            "interpolation_prob": self.interpolation_prob,
            "jpeg_quality": self.jpeg_quality
        }
        label = actual_upscale_factor - 1  # 1x -> 0, 2x -> 1, etc.

        return img_tensor, label, metadata


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    dataset = OptimizedImageProcessingDataset(
        root_dir="./raw_pict", 
        upscale_factor=24, 
        crop_prob=0.1, 
        interpolation_prob=0.1, 
        jpeg_quality=80
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    for img_batch, metadata_batch in dataloader:
        print("Image batch shape:", img_batch.shape)
        print("Metadata:", metadata_batch)
        plt.imshow(transforms.ToPILImage()(img_batch[0]))
        plt.show()
        break
