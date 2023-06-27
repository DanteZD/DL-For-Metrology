import numpy as np
import torch
from Scope import SievesDataset
from torch.utils.data import DataLoader
import cv2 as cv
from keras.metrics import MeanIoU
import os
import csv
import json
import pandas as pd
from matplotlib import pyplot as plt

class MetricTracker:
    def __init__(self, n_classes):
        self.total_mean_iou = MeanIoU(num_classes=n_classes)

    def calculate_iou(self, preds, targets):
        with torch.no_grad():
            preds = torch.sigmoid(preds)
            preds = (preds > 0.5).float()
            preds = preds.cpu().numpy()

            # Loop through an individual batch
            for pred, target in zip(preds, targets):
                target = target.cpu().numpy()
                # Update to the IoU over all images in the batch
                self.total_mean_iou.update_state(target, pred)

    def get(self):
        return self.total_mean_iou.result().numpy()


def save_checkpoint(state, folder, filename):
    print("=> Saving checkpoint")
    print(f"Filename = {filename}")
    model_path = os.path.join(folder, filename)
    torch.save(state, model_path)


def load_checkpoint(model, folder, filename):
    print("=> Loading checkpoint")
    print(f"Filename = {filename}")
    checkpoint_path = os.path.join(folder, filename)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])


def get_loader(
        image_list,
        image_dir,
        mask_dir,
        batch_size,
        transform,
        num_workers=2,
        pin_memory=True,
):
    # Create train dataset + specify the transforms / directories
    ds = SievesDataset(
        image_list,
        image_dir=image_dir,
        mask_dir=mask_dir,
        transform=transform,
    )

    # Create training loader + specify training dataset and parameters + shuffle order randomly
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True
    )

    return loader


def compute_iou_per_class(mean_iou):
    # Matrix: Row = true labels | Columns = predicted labels
    # [0,0] = Number of pixels where, True label = 0 & Predicted label = 0
    # [0,1] = Number of pixels where, True label = 0 & Predicted label = 1
    pred_per_class = np.array(mean_iou.get_weights()[0])

    # IoU per class
    # Sum column and row 0 to get all the TP + FP + FN per class
    # Subtract the diagonal not to double count TP's
    sum_row = np.sum(pred_per_class, axis=1)
    sum_column = np.sum(pred_per_class, axis=0)
    # True Positives
    diagonal = np.diagonal(pred_per_class)
    # Union per class
    union_per_class = sum_row + sum_column - diagonal

    iou_per_class = diagonal/union_per_class

    return iou_per_class


def evaluate_model(loader, model, device="cuda"):
    n_classes = 2
    mean_iou = MeanIoU(num_classes=n_classes)

    # Evaluation mode
    model.eval()

    # Loop over batches
    for x, labels in loader:
        x = x.to(device)

        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

            # Loop through an individual batch
            for pred, true_label in zip(preds, labels):
                pred = pred.cpu().numpy()
                true_label = true_label.cpu().numpy()

                # Update to the IoU over all images in the batch
                mean_iou.update_state(true_label, pred)

    iou_per_class = compute_iou_per_class(mean_iou)

    # Turn training mode back on
    model.train()

    return iou_per_class, mean_iou.result().numpy()


def save_predictions_as_imgs(
        loader, model, batch_size, folder, device="cuda"
):
    # Check if directory exist or create a new
    if not os.path.isdir(folder):
        os.makedirs(folder)

    # Evaluation mode
    model.eval()

    for idx_main, (x, labels), in enumerate(loader):
        x = x.to(device=device)

        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            preds = preds.cpu().numpy()

            # Save images in batch as individual images for inspection
            for idx_inner, (pred, true_label) in enumerate(zip(preds, labels)):
                # Calculate ongoing unique image number
                img_num = idx_main * batch_size + idx_inner

                # Turn class labels to grayscale values for easy inspection
                pred[pred == 1] = 255.0
                # Turn class labels to grayscale values for easy inspection
                true_label[true_label == 1] = 255.0

                # Save predicted segmentations
                cv.imwrite(f"{folder}/pred_{img_num}.png", pred.squeeze(0))
                # Save corresponding true segmentations
                cv.imwrite(f"{folder}/true_{img_num}.png", true_label.numpy())

    # Turn training mode back on
    model.train()


def write_to_csv(csv_path, epoch=None, train_loss=None, train_miou=None,
                 val_miou=None, class_iou=None, create_header=False, test_set=False):
    # Write for test set evaluation
    if test_set:
        header = ["mean_iou", "iou_class0", "iou_class1"]
        with open(csv_path, "w", encoding="UTF8", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(header)
            writer.writerow([val_miou, class_iou[0], class_iou[1]])

        return

    # Create new header for train / validation set
    if create_header:
        header = ["epoch", "train_loss", "train_miou", "val_miou", "val_iou_class0", "val_iou_class1"]
        with open(csv_path, "w", encoding="UTF8", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(header)

        return

    # Add data row to train / validation csv
    with open(csv_path, "a", encoding="UTF8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([epoch, train_loss, train_miou, val_miou, class_iou[0], class_iou[1]])


def save_splits(folder, train, val, test):
    for file, name in zip([train, val, test], ["train", "val", "test"]):
        with open(f"{folder}/{name}_split.json", "w") as f:
            json.dump(file, f, indent=2)


def plot_train_val(model_dir, model_name):
    # Plot training / validation metrics
    # Read in the data into a pandas dataframe
    df = pd.read_csv(f"{model_dir}{model_name}.csv")

    # Training loss
    plt.plot(df["epoch"], df["train_loss"])
    plt.xlabel('Number of Epochs')
    plt.ylabel('Training Loss')
    plt.title("Training Loss per Epoch")
    plt.locator_params(axis="x", integer=True, tight=True)
    plt.show()

    # Training mIoU
    plt.plot(df["epoch"], df["train_miou"])
    plt.plot(df["epoch"], df["val_miou"])
    plt.xlabel('Number of Epochs')
    plt.ylabel('Mean IoU')
    plt.title("Mean IoU per Epoch")
    plt.locator_params(axis="x", integer=True, tight=True)
    plt.legend(["Train", "Validation"])
    plt.show()

    # Validation Class IoU
    plt.plot(df["epoch"], df["val_iou_class0"], color="red")
    plt.plot(df["epoch"], df["val_iou_class1"], color="blue")
    plt.xlabel('Number of Epochs')
    plt.ylabel('Validation Class IoU')
    plt.title("Validation Class IoU per Epoch")
    plt.locator_params(axis="x", integer=True, tight=True)
    plt.legend(["Class 0 (Background)", "Class 1 (Scope)"])
    plt.show()