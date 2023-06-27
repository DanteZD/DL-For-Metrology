"""
Run a single input image through the multiclass
sieve segmentation model for inference.
"""

import os
import cv2 as cv
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import UNET
from Multi.multi_utils import load_checkpoint
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from utils import split_holes_debris

# NO IMAGE DATA IS INCLUDED IN THE GITHUB REPOSITORY
# NO MODEL DATA IS INCLUDED IN THE GITHUB REPOSITORY
# THESE FILES ARE TO0 LARGE OF SIZE
# BEFORE RUNNING THIS SCRIPT FIX THE PATHS FOR YOU FOLDER STRUCTURE!

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_HEIGHT = 720
IMAGE_WIDTH = 720
N_CLASSES = 3

MODEL_DIR = "trained_models/"
# Change to load another model
MODEL_NAME = "sieve_epochs200_batch7_h720_w720_100xlr1e-4_100xlr1e-05_RandomResizedCrops_FocalLoss.pth.tar"

INFER_DIR = "data/"
# Change to load from other folders
IMAGE_DIR = "images/"
# Change to load another image for inference
INPUT_IMAGE = "data-2_63_img75.jpg"


def predict(x, model, device="cuda"):
    # Evaluation mode
    model.eval()
    x = x.to(device)

    with torch.no_grad():
        pred = torch.argmax(model(x), dim=1).squeeze(0)
        pred = pred.cpu().numpy()

        # Turn categorical to grayscale values for easy inspection
        pred[pred == 1] = 127.0
        pred[pred == 2] = 255.0

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
    resized_image = cv.resize(image, (720, 720))

    # Load trained model
    model = UNET(in_channels=3, n_classes=N_CLASSES).to(DEVICE)
    load_checkpoint(model, MODEL_DIR, MODEL_NAME)

    # Add batch dimension
    image_resized = image_resized.clone().detach().unsqueeze(0)

    # Make a prediction
    prediction = predict(image_resized, model)

    # Plot predicted raw segmentation mask
    plt.imshow(prediction, cmap='gray')
    plt.title("Predicted segmentations")
    plt.show()

    # Create visual representation of the segmentation mask
    # Create CSV report
    img_copy = resized_image.copy()
    overlay_img = resized_image.copy()

    prediction = prediction.astype(np.uint8)
    contours, _ = cv.findContours(prediction, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    segment_dict = split_holes_debris(prediction, contours)

    # Loop over all hole segments, measure their dimensions + create segmentation overlay
    for segment in segment_dict['holes']:
        cv.drawContours(img_copy, [segment_dict['holes'][segment].contour], -1, (255, 0, 0), 2)
        overlay_img = cv.fillPoly(overlay_img, pts=[segment_dict['holes'][segment].contour], color=(225, 105, 65),
                                  lineType=cv.LINE_AA)

    # Create segmentation overlay for all debris segments.
    for segment in segment_dict['debris']:
        cv.drawContours(img_copy, [segment_dict['debris'][segment].contour], -1, (226, 43, 138), 2)
        overlay_img = cv.fillPoly(overlay_img, pts=[segment_dict['debris'][segment].contour], color=(226, 43, 138),
                                  lineType=cv.LINE_AA)

    # Opacity of segment overlay
    alpha = 0.4
    output = cv.addWeighted(overlay_img, alpha, img_copy, 1 - alpha, 0)

    plt.imshow(output)
    plt.title("Predicted segments overlaid on the input image")
    plt.show()

    # Save the segmentation overlay image
    # cv.imwrite(f"data/inference/results.png", output)

if __name__ == "__main__":
    main()