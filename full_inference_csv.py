"""
Run a single input image through the full inference pipeline.
The resulting output is a CSV file containing the dimensions of all
hole segments that are within the region of interest.
Additionally, a visual representation is created for inspection.
"""

import os
import numpy as np
import cv2 as cv
from PIL import Image
from matplotlib import pyplot as plt
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import UNET
from Multi.multi_utils import (
    load_checkpoint,
)
from utils.post_proc_utils import(
    micron_to_pixel_ratio,
    split_holes_debris,
    cal_avg_area_width_height,
    post_process,
    write_csv_report
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_CLASSES = 3
IMAGE_HEIGHT = 720
IMAGE_WIDTH = 720

# NO IMAGE DATA IS INCLUDED IN THE GITHUB REPOSITORY
# NO MODEL DATA IS INCLUDED IN THE GITHUB REPOSITORY
# THESE FILES ARE TO LARGE OF SIZE
# BEFORE RUNNING THIS SCRIPT FIX THE PATHS FOR YOU FOLDER STRUCTURE!

MODEL_DIR = "trained_models/"

MODEL_NAME_SIEVE = "sieve_epochs200_batch7_h720_w720_100xlr1e-4_100xlr1e-05_RandomResizedCrops_FocalLoss.pth.tar"
MODEL_NAME_SCOPE = "scope_epochs100_batch7_h720_w720_50xlr1e-04_50xlr1e-05_RandomResizedCrops.pth.tar"

IMAGE_DIR = "data/inference/original_images/data-2/63/"
INPUT_IMG = "PXL_20230220_122318396.MP.jpg"


SAVE_DIR = "data/inference/saved/"
CSV_DIR = "data/inference/csv/"


def predict(x, model_sieve, model_scope, device="cuda"):
    # Evaluation mode
    model_sieve.eval()
    model_scope.eval()

    x = x.to(device)

    with torch.no_grad():
        # Make prediction for the sieve module
        pred_sieve = torch.argmax(model_sieve(x), dim=1).squeeze(0)
        pred_sieve = pred_sieve.cpu().numpy()

        # Make prediction for the lens module
        pred_scope = torch.sigmoid(model_scope(x))
        pred_scope = (pred_scope > 0.5).float().squeeze(0, 1)
        pred_scope = pred_scope.cpu().numpy()

        pred_sieve[pred_sieve == 1] = 127
        pred_sieve[pred_sieve == 2] = 255
        pred_sieve = pred_sieve.astype(np.uint8)

        pred_scope[pred_scope == 1] = 255
        pred_scope = pred_scope.astype(np.uint8)

    # Turn training mode back on
    model_sieve.train()
    model_scope.train()

    return pred_sieve, pred_scope


def main():
    img_path = os.path.join(IMAGE_DIR, INPUT_IMG)
    img = np.array(Image.open(img_path).convert("RGB"))

    img_height, img_width = img.shape[:2]

    # Resize (+ normalize) the image to the trained model HxW
    transform = A.Compose(
        [
            A.Resize(height=720, width=720),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ]
    )

    img_resized = transform(image=img)
    img_resized = img_resized['image']

    # Add batch dimension
    img_resized = img_resized.clone().detach().unsqueeze(0)

    # Load trained model SIEVE
    model_sieve = UNET(in_channels=3, n_classes=N_CLASSES).to(DEVICE)
    load_checkpoint(model_sieve, MODEL_DIR, MODEL_NAME_SIEVE)

    # Load trained model SCOPE
    model_scope = UNET(in_channels=3, n_classes=1).to(DEVICE)
    load_checkpoint(model_scope, MODEL_DIR, MODEL_NAME_SCOPE)

    # Predict segmentation masks
    pred_sieve, pred_scope = predict(img_resized, model_sieve, model_scope)

    # Apply post-processing procedure
    post_pred_sieve, scope_mask, scaled_scope_mask = post_process(pred_sieve, pred_scope, eval=False)

    # Calculate the micron to pixel ratio for this image
    micron_pixel_ratio = micron_to_pixel_ratio(scope_mask, img_width, img_height)

    transform_resize = A.Resize(width=img_width, height=img_height)
    resized_prediction= transform_resize(image=post_pred_sieve)
    resized_prediction = resized_prediction["image"]

    # Create visual representation of the segmentation mask
    # Create CSV report
    img_copy = img.copy()
    overlay_img = img.copy()

    segment_contours, _ = cv.findContours(resized_prediction, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    segment_dict = split_holes_debris(resized_prediction, segment_contours)

    # CSV Report -> First 2 lines
    m_area, m_width, m_height = cal_avg_area_width_height(segment_dict['holes'])
    m_area *= micron_pixel_ratio
    m_width *= micron_pixel_ratio
    m_height *= micron_pixel_ratio
    number_debris = len(segment_dict['debris'])

    csv_name = INPUT_IMG.replace(".MP.jpg", ".csv")
    csv_path = os.path.join(CSV_DIR, f"{csv_name}")
    write_csv_report(csv_path, number_debris, m_area, m_width, m_height, create_header1=True)
    write_csv_report(csv_path, create_header2=True)

    # Loop over all hole segments, measure their dimensions + create segmentation overlay
    for segment in segment_dict['holes']:
        cv.drawContours(img_copy, [segment_dict['holes'][segment].contour], -1, (255, 0, 0), 8)
        overlay_img = cv.fillPoly(overlay_img, pts=[segment_dict['holes'][segment].contour], color=(225, 105, 65),
                                  lineType=cv.LINE_AA)

        # Add the individual dimensions of every hole in microns to the csv report
        area = segment_dict['holes'][segment].area * micron_pixel_ratio
        width = segment_dict['holes'][segment].width * micron_pixel_ratio
        height = segment_dict['holes'][segment].height * micron_pixel_ratio
        cont = segment_dict['holes'][segment].contour
        # Calculate the moments of the contour
        M = cv.moments(cont)
        # Calculate the center of the contour
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        center = [cx, cy]

        write_csv_report(csv_path, center, area, width, height)

    # Create segmentation overlay for all debris segments.
    for segment in segment_dict['debris']:
        cv.drawContours(img_copy, [segment_dict['debris'][segment].contour], -1, (226, 43, 138), 6)
        overlay_img = cv.fillPoly(overlay_img, pts=[segment_dict['debris'][segment].contour], color=(226, 43, 138),
                                  lineType=cv.LINE_AA)

    # Opacity of segment overlay
    alpha = 0.3
    output = cv.addWeighted(overlay_img, alpha, img_copy, 1 - alpha, 0)

    # Resize output segmentation mask specify wanted dimensions
    output = cv.resize(output, (956, 720))

    plt.imshow(output)
    plt.title("Resulting segmentation mask after post-processing")
    plt.show()

    # Save visual representation of the overlay for inspection.
    img_name = INPUT_IMG.replace(".MP.jpg", ".png")
    cv.imwrite(f"{SAVE_DIR}/pred_{img_name}", output)

if __name__ == "__main__":
    main()