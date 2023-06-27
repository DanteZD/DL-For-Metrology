"""
This file can be utilized for evaluating a model on it's test set
after the post-processing procedure.
"""

import numpy as np
from tqdm import tqdm
from keras.metrics import MeanIoU
import os
import json
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import UNET
from Multi.multi_utils import (
    load_checkpoint,
    get_loader,
    compute_iou_per_class
)
from utils.post_proc_utils import post_process

# NO IMAGE DATA IS INCLUDED IN THE GITHUB REPOSITORY
# NO MODEL DATA IS INCLUDED IN THE GITHUB REPOSITORY
# THESE FILES ARE TO LARGE OF SIZE
# BEFORE RUNNING THIS SCRIPT FIX THE PATHS FOR YOU FOLDER STRUCTURE!

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
NUM_WORKERS = 2
N_CLASSES = 3
IMAGE_HEIGHT = 720
IMAGE_WIDTH = 720
PIN_MEMORY = True
IMAGE_DIR = "data/images/"
MASK_DIR = "data/sieve_masks/"
MODEL_DIR = "trained_models/"
FILE_NAME = "test_split.json"

MODEL_NAME_SIEVE = "sieve_epochs200_batch7_h720_w720_100xlr1e-4_100xlr1e-05_RandomResizedCrops_FocalLoss"
MODEL_NAME_SCOPE = "scope_epochs100_batch7_h720_w720_50xlr1e-04_50xlr1e-05_RandomResizedCrops"


def evaluate_model(loader, model_sieve, model_scope, device="cuda"):
    loop = tqdm(loader)
    loop.set_description(f"Batch Progress")
    n_classes = 3
    mean_iou = MeanIoU(num_classes=n_classes)

    # Evaluation mode
    model_sieve.eval()
    model_scope.eval()

    # Loop over batches
    for batch_idx, (x, labels) in enumerate(loop):
        x = x.to(device)

        with torch.no_grad():
            # Make prediction for the sieve module
            preds_sieve = torch.argmax(model_sieve(x), dim=1)
            preds_sieve = preds_sieve.cpu().numpy()

            # Make prediction for the lens module
            preds_scope = torch.sigmoid(model_scope(x))
            preds_scope = (preds_scope > 0.5).float()
            preds_scope = preds_scope.cpu().numpy()

            # Loop through an individual batch
            for pred_sieve, pred_scope, true_label in zip(preds_sieve, preds_scope, labels):

                pred_scope = pred_scope.squeeze(0)
                true_label = true_label.cpu().numpy()

                pred_sieve[pred_sieve == 1] = 127
                pred_sieve[pred_sieve == 2] = 255
                pred_sieve = pred_sieve.astype(np.uint8)

                pred_scope[pred_scope == 1] = 255
                pred_scope = pred_scope.astype(np.uint8)

                true_label[true_label == 1] = 127
                true_label[true_label == 2] = 255
                true_label = true_label.astype(np.uint8)

                new_mask, new_true = post_process(pred_sieve, pred_scope, true_label, eval=True)

                # Turn image pixels into categorical values for evaluation
                new_mask[new_mask == 127] = 1
                new_mask[new_mask == 255] = 2

                new_true[new_true == 127] = 1
                new_true[new_true == 255] = 2

                # Update to the IoU over all images in the batch
                mean_iou.update_state(new_true, new_mask)

    iou_per_class = compute_iou_per_class(mean_iou)

    # Turn training mode back on
    model_sieve.train()
    model_scope.train()

    return iou_per_class, mean_iou.result().numpy()


def main():
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
    json_path = os.path.join(MODEL_DIR, FILE_NAME)
    with open(json_path) as f:
        test_json = f.read()
    # Create split list
    test_split = json.loads(test_json)

    # Load trained model SIEVE
    model_sieve = UNET(in_channels=3, n_classes=N_CLASSES).to(DEVICE)
    load_checkpoint(model_sieve, MODEL_DIR, f"{MODEL_NAME_SIEVE}.pth.tar")

    # Load trained model SCOPE
    model_scope = UNET(in_channels=3, n_classes=1).to(DEVICE)
    load_checkpoint(model_scope, MODEL_DIR, f"{MODEL_NAME_SCOPE}.pth.tar")

    # Create test ds + loader
    test_loader = get_loader(test_split, IMAGE_DIR, MASK_DIR, BATCH_SIZE,
                             transform, NUM_WORKERS, PIN_MEMORY)

    # Evaluate mIoU and class IoU
    class_iou, miou = evaluate_model(test_loader, model_sieve, model_scope, device=DEVICE)

    print(f"Class iou: {class_iou}")
    print(f"mean iou: {miou}")

if __name__ == "__main__":
    main()