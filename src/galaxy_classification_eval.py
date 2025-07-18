"""
Galaxy Classification Model Evaluation Script
=============================================

This script evaluates a trained convolutional neural network (GalaxyCNN) on a
galaxy image classification task. It supports loading multiple model checkpoints
from a directory, computing accuracy and confusion matrix metrics, visualising
feature maps, and saving misclassified examples.

Key Features:
-------------
- Loads grayscale test images from a folder structured by class.
- Evaluates model checkpoints (.pt or .pth) using test accuracy and per-class metrics.
- Saves accuracy logs to CSV.
- Visualises intermediate feature maps from early convolutional layers.
- Optionally copies misclassified images into a structured results folder.

Directory Expectations:
-----------------------
- `test_data/` should contain subdirectories per class with test images.
- `model_trained/` should contain PyTorch model checkpoints.
- `Results/` is used for saving evaluation outputs like accuracy logs.

Usage:
------
- Set `USE_DEFAULT_PATHS` to True/False to toggle between fixed or manual path configuration.
- Run as a script:
    $ python galaxy_classification_eval_final.py

Dependencies:
-------------
- torch
- torchvision
- numpy
- matplotlib
- seaborn
- scikit-learn
- statsmodels
"""

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from collections import defaultdict

import torch.nn as nn
import matplotlib.pyplot as plt
import torch
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
import shutil


class GalaxyCNN(nn.Module):
    """Convolutional Neural Network for grayscale galaxy classification."""
    def __init__(self, num_classes):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * 2 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

def build_transform():
    """Builds the image transformation pipeline for preprocessing test data."""
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

def model_update(net, path):
    """Loads weights from a checkpoint file into the given model."""
    checkpoint = torch.load(path, weights_only=True)
    net.load_state_dict(checkpoint['state_dict'])

def find_classification(path_dir):
    """Returns number of classes and list of class names from a directory."""
    folders = [f for f in os.listdir(path_dir) if os.path.isdir(os.path.join(path_dir, f))]
    return len(folders), folders

def data_loader(root_test, transform):
    """Creates a DataLoader for the test dataset."""
    dataset_test = ImageFolder(root=root_test, transform=transform)
    test_batch_size = 12
    loader_test = DataLoader(dataset_test, batch_size=test_batch_size, shuffle=False)
    return loader_test

def test(model, loader, criterion, device):
    """Evaluates the model on test data and returns metrics and misclassified samples."""
    model.eval()
    test_loss = 0
    correct = 0
    all_preds = []
    all_labels = []
    wrong_samples = []

    dataset = loader.dataset
    assert hasattr(dataset, "samples"), "Expected ImageFolder dataset"
    sample_idx = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            for i in range(len(images)):
                pred = preds[i].item()
                true = labels[i].item()
                filepath, _ = dataset.samples[sample_idx]
                if pred != true:
                    wrong_samples.append((filepath, true, pred))
                sample_idx += 1

            correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss /= len(loader)
    accuracy = correct / len(loader.dataset)
    cm = confusion_matrix(all_labels, all_preds)
    return test_loss, accuracy, cm, wrong_samples

def save_wrong_predictions(wrong_samples, class_names, output_root="wrong_predictions", base_data_root=None):
    """Copies misclassified test images into class-wise subfolders for analysis."""
    for filepath, true_idx, pred_idx in wrong_samples:
        true_class = class_names[true_idx]
        pred_class = class_names[pred_idx]
        rel_path = os.path.relpath(filepath, base_data_root) if base_data_root else filepath
        sub_path = rel_path.split(os.sep, 1)[-1]
        dest_path = os.path.join(output_root, pred_class, sub_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy2(filepath, dest_path)

def save_accuracies(root_result, base_name, accuracy_list):
    """Saves model accuracy values to a CSV file."""
    filename = os.path.join(root_result, f"{base_name}_log.csv")
    header = "accuracy"
    np.savetxt(filename, accuracy_list, delimiter=",", header=header, comments='', fmt="%.6f")
    print("accuracies saved")

def plot_confusion_matrix(cm, class_names):
    """Displays a heatmap of the confusion matrix."""
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Disk vs Non-Disk Confusion Matrix")
    plt.tight_layout()
    plt.show()

def print_per_class_accuracy(cm, class_names=None):
    """Prints accuracy for each class and overall accuracy."""
    total_correct = 0
    total_samples = 0
    for i in range(len(cm)):
        correct = cm[i][i]
        total = cm[i].sum()
        acc = correct / total if total > 0 else 0.0
        name = class_names[i] if class_names else f"Class {i}"
        print(f"{name}: {acc * 100:.2f}%")
        total_correct += correct
        total_samples += total
    total_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    print(f"Total Accuracy: {total_accuracy * 100:.2f}%\n")

def visualise_feature_map(model, image_tensor, layer_idx, grayscale=True):
    """Visualises feature maps from a specified convolutional layer."""
    model.eval()
    with torch.no_grad():
        x = image_tensor.to(next(model.parameters()).device)
        for i, layer in enumerate(model.conv_layers):
            x = layer(x)
            if i == layer_idx:
                break
    feature_maps = x.squeeze(0).cpu()
    n_maps = feature_maps.shape[0]
    cols = 8
    rows = (n_maps + cols - 1) // cols
    plt.figure(figsize=(cols * 2, rows * 2))
    for i in range(n_maps):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(feature_maps[i], cmap='gray' if grayscale else None)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_one_featuremap(model, loader):
    """Visualises the first feature map from the first test sample."""
    images, labels = next(iter(loader))
    visualise_feature_map(model, images[0].unsqueeze(0), layer_idx=0)

def main():
    """Main execution: loads test data, evaluates models, saves and plots results."""
    base_name = 'example_model'
    USE_DEFAULT_PATHS = True
    if USE_DEFAULT_PATHS:
        root_test = os.path.normpath(os.path.join("..", "data_preparation", "dataset", "test_data"))
        checkpoint_root = os.path.normpath(os.path.join("..", "model_trained"))
        root_result = os.path.normpath(os.path.join("..", "Results", "accuracy_results"))
    else:
        root_test = r'path/to/test'
        checkpoint_root = r'root/to/test'
        root_result = r'path/to/results'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(torch.cuda.get_device_name(0))
    transform = build_transform()
    num_classes, classes = find_classification(root_test)
    loader_test = data_loader(root_test, transform)
    class_names = loader_test.dataset.classes
    model = GalaxyCNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    accuracy_list = []

    for checkpoint_file in os.listdir(checkpoint_root):
        if checkpoint_file.endswith(".pt") or checkpoint_file.endswith(".pth"):
            print(checkpoint_file)
            checkpoint_path = os.path.join(checkpoint_root, checkpoint_file)
            model_update(model, checkpoint_path)
            test_loss, test_accuracy, cm, wrong_samples = test(model, loader_test, criterion, device)
            print_per_class_accuracy(cm, class_names)
            accuracy_list.append(test_accuracy)

    save_accuracies(root_result, base_name, accuracy_list)

    #Plot confusion matrix and feature map of the last model in the loop
    #Edit this part to print other model manually
    plot_one_featuremap(model, loader_test)
    plot_confusion_matrix(cm, class_names)

    #It saves wrongly classified images by model
    #save_wrong_predictions(wrong_samples, class_names, base_data_root=loader_test.dataset.root)

if __name__ == "__main__":
    main()
