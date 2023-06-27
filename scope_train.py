"""
Train the binary lens segmentation U-Net model.
"""

import os.path
import numpy as np
import random
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from Scope.scope_utils import (
    MetricTracker,
    load_checkpoint,
    save_checkpoint,
    get_loader,
    evaluate_model,
    save_predictions_as_imgs,
    write_to_csv,
    save_splits,
    plot_train_val
)

# NO IMAGE DATA IS INCLUDED IN THE GITHUB REPOSITORY
# NO MODEL DATA IS INCLUDED IN THE GITHUB REPOSITORY
# NO GROUND TRUTH MASKS ARE INCLUDED
# THESE FILES ARE TO0 LARGE OF SIZE FOR GITHUB
# BEFORE RUNNING THIS SCRIPT FIX THE PATHS FOR YOU FOLDER STRUCTURE!

# Hyper parameters & settings
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 7
NUM_EPOCHS = 2
NUM_WORKERS = 2
IMAGE_HEIGHT = 720
IMAGE_WIDTH = 720
PIN_MEMORY = True
LOAD_MODEL = False
IMAGE_DIR = "data/images/"
MASK_DIR = "data/scope_masks/"
SAVED_IMAGES_DIR = "saved_images/"
MODEL_DIR = 'saved_model_scope/'
MODEL_NAME = f'epochs{NUM_EPOCHS}_batch{BATCH_SIZE}_h{IMAGE_HEIGHT}_w{IMAGE_WIDTH}_lr{LEARNING_RATE}_RandomResizedCrops'
N_CLASSES = 1
TRAIN_RATIO = 0.8               # 80%
VAL_RATIO = TRAIN_RATIO + 0.1   # 10% | Remaining % -> 10% is for the test set


def train_fn(loader, model, optimizer, loss_fn, scaler, epoch):
    loop = tqdm(loader)
    epoch_metrics = MetricTracker(2)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

            # Full epoch metrics: cumulative metrics of the batches
            epoch_metrics.calculate_iou(predictions, targets)

        # backwards
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_description(f"Epoch: {epoch}/{NUM_EPOCHS}")
        loop.set_postfix(loss=loss.item())

    return loss.item(), epoch_metrics.get()

def main():
    # Check if directory exist or create a new
    if not os.path.isdir(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    train_transform = A.Compose(
        [
            A.RandomResizedCrop(width=IMAGE_WIDTH, height=IMAGE_HEIGHT, scale=(0.1, 1.0), p=1.0),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ]
    )

    val_transform = A.Compose(
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

    # Create train / val / test split randomly
    all_images = os.listdir(IMAGE_DIR)
    random.seed(42)
    random.shuffle(all_images)
    train_split, val_split, test_split = np.split(
        all_images,
        [int(TRAIN_RATIO * len(all_images)), int(VAL_RATIO * len(all_images))]
    )

    # Save splits for later evaluation -> especially the test split for this trained model
    save_splits(MODEL_DIR, list(train_split), list(val_split), list(test_split))

    train_loader = get_loader(train_split, IMAGE_DIR, MASK_DIR, BATCH_SIZE,
                              train_transform, NUM_WORKERS, PIN_MEMORY
                              )

    val_loader = get_loader(val_split, IMAGE_DIR, MASK_DIR, BATCH_SIZE,
                            val_transform, NUM_WORKERS, PIN_MEMORY
                            )

    print(
        f"Training for: {NUM_EPOCHS} epochs | Batchsize : {BATCH_SIZE} | WxH: {IMAGE_WIDTH}x{IMAGE_HEIGHT} | LR: {LEARNING_RATE}")

    model = UNET(in_channels=3, n_classes=N_CLASSES).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Load model for further training
    if LOAD_MODEL:
        load_checkpoint(model, f"model path", f"model name")

    # Gradient scaling, inorder to prevent vanishing or exploding gradients
    scaler = torch.cuda.amp.GradScaler()

    # Dummy variable model performance comparison
    # Holding the epoch & Mean_IoU of when the model performed best
    best_mean_iou = (0, -1)

    csv_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}.csv")
    write_to_csv(csv_path, create_header=True)

    # Training Loop
    for epoch in range(NUM_EPOCHS):
        # Train model and return the loss
        train_loss, train_miou = train_fn(train_loader, model, optimizer, loss_fn, scaler, epoch + 1)

        # Save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        # Evaluate the current training epoch on the validation data
        val_class_iou, val_miou = evaluate_model(val_loader, model, device=DEVICE)
        write_to_csv(csv_path, epoch + 1, train_loss, train_miou, val_miou, val_class_iou)

        print(f"Train mIoU: {train_miou} | Validation mIoU: {val_miou} | Validation Class IoU: {val_class_iou}")

        if val_miou > best_mean_iou[1]:
            print(
                f"New best validation mIoU: {val_miou} | Previous best mIoU: {best_mean_iou[1]} in epoch: {best_mean_iou[0]}.")
            # save_checkpoint(checkpoint, folder=f"saved_models/", filename=f"checkpoint_epoch{epoch+1}.pth.tar")
            # For now just overwrite previous best model
            save_checkpoint(checkpoint, folder=MODEL_DIR, filename=f"{MODEL_NAME}.pth.tar")

            # Save some examples to a folder for inspection
            save_predictions_as_imgs(
                val_loader, model, BATCH_SIZE, folder=f"{MODEL_DIR}{SAVED_IMAGES_DIR}epoch{epoch + 1}/", device=DEVICE
            )

            # Update best IoU
            best_mean_iou = (epoch + 1, val_miou)

    plot_train_val(MODEL_DIR, MODEL_NAME)


if __name__ == "__main__":
    main()

