import os
import argparse
from pathlib import Path
import torch
from model import PSDN
import PIL.Image as Image
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from PIL import ImageSequence

def load_model(model_path):
    model = PSDN(n_pixel_class=16)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_pixel_size(model, image):
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        result = model.predict(image_tensor)
        return result.item() + 1

def process_image(image_path, model, output_dir, dest_pixel_size=1):
    image = Image.open(image_path)
    if image.format == 'GIF':
        return process_gif(image, model, output_dir, image_path.name, dest_pixel_size)
    else:
        return process_static_image(image, model, output_dir, image_path.name, dest_pixel_size)

def process_static_image(image, model, output_dir, file_name, dest_pixel_size=1):
    detect_image = image.convert("L")
    pixel_size = predict_pixel_size(model, detect_image)
    resized_image = image.resize(
        (image.width // pixel_size, image.height // pixel_size),
        resample=Image.NEAREST
    )
    resized_image = resized_image.resize(
        (resized_image.width * dest_pixel_size, resized_image.height * dest_pixel_size),
        resample=Image.NEAREST
    )
    output_path = output_dir / file_name
    resized_image.save(output_path)
    print(f"Processed {file_name}: Pixel size {pixel_size}, output saved to {output_path}")

def process_gif(gif, model, output_dir, file_name, dest_pixel_size=1):
    frames = []
    pixel_size = predict_pixel_size(model, gif.convert("L"))
    for frame in ImageSequence.Iterator(gif):
        resized_frame = frame.resize(
            (frame.width // pixel_size, frame.height // pixel_size),
            resample=Image.NEAREST
        )
        resized_frame = resized_frame.resize(
            (resized_frame.width * dest_pixel_size, resized_frame.height * dest_pixel_size),
            resample=Image.NEAREST
        )
        frames.append(resized_frame)

    output_path = output_dir / file_name
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        loop=gif.info.get('loop', 0),
        duration=gif.info.get('duration', 100)
    )
    print(f"Processed {file_name}: Pixel size {pixel_size}, output saved to {output_path}")

def process_directory(directory_path, model, output_dir):
    for file in Path(directory_path).iterdir():
        if file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif']:
            process_image(file, model, output_dir)

def main():
    parser = argparse.ArgumentParser(description="Image downsampling tool based on predicted pixel size.")
    parser.add_argument("input", type=str, help="Path to an image, GIF, or directory containing images.")
    parser.add_argument("--model", type=str, default="./psdn_model.pth", help="Path to the trained model file.")
    parser.add_argument("--output", type=str, default="./output", help="Output directory for processed images.")
    parser.add_argument("--dest_pixel_size", type=int, default=1, help="Destination pixel size for downscaling.")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.model)

    dest_pixel_size = args.dest_pixel_size

    if input_path.is_dir():
        process_directory(input_path, model, output_dir, dest_pixel_size)
    elif input_path.is_file():
        process_image(input_path, model, output_dir, dest_pixel_size)
    else:
        print(f"Invalid input path: {args.input}")

if __name__ == "__main__":
    main()
