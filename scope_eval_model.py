"""
Evaluate the binary lens segmentation model
"""

import os
import json
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import UNET
from Scope.scope_utils import (
    load_checkpoint,
    get_loader,
    evaluate_model,
    save_predictions_as_imgs,
    write_to_csv,
)

# NO IMAGE DATA IS INCLUDED IN THE GITHUB REPOSITORY
# NO MODEL DATA IS INCLUDED IN THE GITHUB REPOSITORY
# THESE FILES ARE T0O LARGE OF SIZE
# BEFORE RUNNING THIS SCRIPT FIX THE PATHS FOR YOU FOLDER STRUCTURE!

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
NUM_WORKERS = 2
N_CLASSES = 1
IMAGE_HEIGHT = 720
IMAGE_WIDTH = 720
PIN_MEMORY = True
IMAGE_DIR = "data/images/"
MASK_DIR = "data/scope_masks/"
SAVED_IMAGES_DIR = "saved_model_multi/saved_images/"
MODEL_DIR = "saved_model_scope/"
TEST_DIR = "eval_test/"

# Change to load another model
MODEL_NAME = "epochs100_batch7_h720_w720_50xlr1e-04_50xlr1e-05_RandomResizedCrops"
# Change to read from another data split
FILE_NAME = "test_split.json"

def main():
    # Trying with transform first
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

    # Retrieve test split for this model
    with open(f"{MODEL_DIR}{FILE_NAME}") as f:
        test_json = f.read()
    # Create split list
    test_split = json.loads(test_json)

    # Load trained model
    model = UNET(in_channels=3, n_classes=N_CLASSES).to(DEVICE)
    load_checkpoint(model, MODEL_DIR, f"{MODEL_NAME}.pth.tar")

    # Create test ds + loader
    test_loader = get_loader(test_split, IMAGE_DIR, MASK_DIR, BATCH_SIZE,
                             transform, NUM_WORKERS, PIN_MEMORY)

    # Evaluate mIoU and class IoU
    class_iou, miou = evaluate_model(test_loader, model, device=DEVICE)
    print(
        f"----------------------------------------------------------------------------------------------------------------")
    print(f"Evaluating model on the test set | mIoU: {miou}")
    print(f"IoU class 0 (background): {class_iou[0]} | IoU class 1 (scope): {class_iou[1]}")
    print(
        f"----------------------------------------------------------------------------------------------------------------")

    # Save metrics
    csv_path = os.path.join(MODEL_DIR, TEST_DIR, f"{MODEL_NAME}.csv")
    write_to_csv(csv_path, val_miou=miou, class_iou=class_iou, test_set=True)

    # Save predictions and masks for inspection
    save_path = os.path.join(MODEL_DIR, TEST_DIR, "Examples/")
    save_predictions_as_imgs(
        test_loader, model, BATCH_SIZE, folder=save_path, device=DEVICE
    )


if __name__ == "__main__":
    main()