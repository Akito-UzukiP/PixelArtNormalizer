import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
from tqdm import tqdm
from image_dataset import OptimizedImageProcessingDataset
from model import PSDN

# Training function
def train_model(model, dataloader, criterion, optimizer, num_epochs=10, device="cuda"):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        tqdm_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for images, gt, metadata in tqdm_bar:
            images = images.to(device)
            labels = gt.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            tqdm_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    print("Training complete.")

# Evaluation function
def evaluate_model(model, dataloader, device="cuda"):
    model.to(device)
    model.eval()
    with torch.no_grad():
        for images, gt, _ in dataloader:
            images = images.to(device)
            outputs = model.predict(images)
            print("Predicted Labels:", outputs.cpu().numpy())
            # Display first image for visualization
            plt.imshow(ToPILImage()(images[0].cpu()))
            plt.title(f"Predicted: {outputs[0].item()}")
            plt.show()
            break

if __name__ == "__main__":
    # Dataset and DataLoader
    dataset = OptimizedImageProcessingDataset(root_dir="./raw_pict", upscale_factor=16, crop_prob=0.1, interpolation_prob=0.3, jpeg_quality=80)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

    # Model
    model = PSDN(n_pixel_class=16)

    model.load_state_dict(torch.load("psdn_model.pth"))

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, dataloader, criterion, optimizer, num_epochs=10)

    torch.save(model.state_dict(), "psdn_model.pth")

    # Evaluate the model
    evaluate_model(model, dataloader)

    # Save the model
