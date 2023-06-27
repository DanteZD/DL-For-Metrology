import os
import cv2 as cv
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import UNET
from Scope.scope_utils import load_checkpoint
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

# NO IMAGE DATA IS INCLUDED IN THE GITHUB REPOSITORY
# NO MODEL DATA IS INCLUDED IN THE GITHUB REPOSITORY
# THESE FILES ARE T0O LARGE OF SIZE
# BEFORE RUNNING THIS SCRIPT FIX THE PATHS FOR YOU FOLDER STRUCTURE!

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_HEIGHT = 720
IMAGE_WIDTH = 720
N_CLASSES = 1

MODEL_DIR = "trained_models/"
# Change to load another model
MODEL_NAME = "scope_epochs100_batch7_h720_w720_50xlr1e-04_50xlr1e-05_RandomResizedCrops.pth.tar"

INFER_DIR = "data/inference/"
# Change to load from other folders
IMAGE_DIR = "original_images/data-7/150/"
# Change to load another image for inference
INPUT_IMAGE = "PXL_20230220_120252330.MP.jpg"

def predict(x, model, device="cuda"):
    # Evaluation mode
    model.eval()
    x = x.to(device)

    with torch.no_grad():
        pred = torch.sigmoid(model(x))
        # Convert to binary image and remove batch, channel dimensions
        pred = (pred > 0.5).float().squeeze(0, 1)
        pred = pred.cpu().numpy()

        # Turn categorical to grayscale values for easy inspection
        pred[pred == 1] = 255.0

    # Turn training mode back on
    model.train()

    return pred

def main():
    img_path = os.path.join(INFER_DIR, IMAGE_DIR, INPUT_IMAGE)
    image = np.array(Image.open(img_path).convert("RGB"))

    # Save original image size
    og_height, og_width, _ = image.shape

    # Resize + normalize the image to the trained model HxW
    transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ]
    )

    resized = transform(image=image)
    image_resized = resized["image"]

    # Load trained model
    model = UNET(in_channels=3, n_classes=N_CLASSES).to(DEVICE)
    load_checkpoint(model, MODEL_DIR, MODEL_NAME)

    # Add batch dimension
    image_resized = image_resized.clone().detach().unsqueeze(0)

    # Make a prediction
    prediction = predict(image_resized, model)

    plt.imshow(prediction, cmap='gray')
    plt.title("Predicted scope segment")
    plt.show()

    # Save predicted segmentation mask
    # cv.imwrite(f"{INFER_DIR}/scope_pred_inference.png", prediction)


if __name__ == "__main__":
    main()