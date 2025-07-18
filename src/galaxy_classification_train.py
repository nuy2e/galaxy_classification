"""
Galaxy Morphology Classification Using CNNs
------------------------------------------------
This script trains a convolutional neural network to classify galaxies from SDSS images
into morphological classes using Galaxy Zoo 2 annotations. It supports weighted loss,
checkpointing, and visualisation of performance over epochs.
"""

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset

import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import torch

import numpy as np
import os

class GalaxyCNN(nn.Module):
    """
    Deep CNN model with 7 convolutional layers followed by 2 fully connected layers.
    Designed for grayscale galaxy image classification.
    """
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

def build_transform_train():
    """
    Builds training data transform: grayscale, crop, normalize.
    Add any augmentation needed before crop.
    """
    return(transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.CenterCrop((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ]))

def build_transform_val():
    """Builds validation data transform: grayscale, crop, normalize."""
    return(transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.CenterCrop((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ]))

def build_criterion(class_dir, device):
    """
    Constructs a weighted cross-entropy loss function based on inverse class frequencies.
    """
    class_counts = {class_name: len([
    f for root, _, files in os.walk(os.path.join(class_dir, class_name))
    for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))
]) for class_name in os.listdir(class_dir)}

    counts = list(class_counts.values())
    total = sum(counts)
    weights = [total / c for c in counts]  # inverse frequency
    normed_weights = [w / sum(weights) for w in weights]  # normalize if desired
    class_weights = torch.tensor(normed_weights, dtype=torch.float32).to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    return criterion

def data_loader(root_train, root_val, transform_train, transform_val, subset_size):
    """
    Loads and optionally subsets training/validation datasets.
    It divides the validation dataset into the same number of batches as the training dataset.
    Therefore, if the validation dataset is too small or the batch size is too large,
    some validation batches may end up empty, which will cause an error in the train_val() function.
    In that case, try increasing the batchsize.
    Returns loaders for training and validation batches.

    """
    dataset_train = ImageFolder(root = root_train, transform = transform_train)
    dataset_val = ImageFolder(root = root_val, transform = transform_val)

    if subset_size:
        random_indices_train = random.sample(range(len(dataset_train)), subset_size)
        random_indices_val = random.sample(range(len(dataset_val)), int(subset_size / 8))
        dataset_train = Subset(dataset_train, random_indices_train)
        dataset_val= Subset(dataset_val, random_indices_val)

    train_batch_size = 24
    loader_train = DataLoader(dataset_train, batch_size= train_batch_size, shuffle=True)#, num_workers=4, pin_memory=True)
    num_batches = len(loader_train)

    # 2. Manually split validation dataset into equal-ish chunks
    val_indices = np.array_split(np.arange(len(dataset_val)), num_batches)
    val_subsets = [Subset(dataset_val, idx.tolist()) for idx in val_indices]

    # 3. Create a list of DataLoaders, one per batch
    loader_val = [DataLoader(subset, batch_size=len(subset)) for subset in val_subsets]

    return loader_train, loader_val

def find_classification(path_dir):
    """Returns the number and names of class folders in the given path."""
    path = path_dir  # Replace with your path
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

    return (len(folders)), folders

def train_val(model, loader_train, loader_val, optimizer, scheduler, criterion, epoch, device):
    """
    Runs one epoch of training and batch-aligned validation.
    Returns average loss and accuracy for both.
    """

    total_train_loss = 0
    correct_train = 0
    total_number_train = 0

    total_val_loss = 0
    correct_val = 0
    total_number_val = 0

    for batch_idx, (images_train, labels_train) in enumerate(loader_train):
        model.train()

        images_train, labels_train = images_train.to(device), labels_train.to(device)
        optimizer.zero_grad()
        output_train = model(images_train)
        loss_train = criterion(output_train, labels_train)
        loss_train.backward()
        optimizer.step()

        #train results
        total_train_loss += loss_train.item()
        preds_train = output_train.argmax(dim=1)
        correct_train += (preds_train == labels_train).sum().item()
        total_number_train += labels_train.size(0)

        print(f"\rTrain Epoch {epoch} [{(batch_idx + 1) * len(images_train)}/{len(loader_train.dataset)}] Loss: {loss_train.item():.4f}", end='')

        with torch.no_grad():
            model.eval()

            images_val, labels_val = next(iter(loader_val[batch_idx]))
            images_val, labels_val = images_val.to(device), labels_val.to(device)
            output_val = model(images_val)
            loss_val = criterion(output_val, labels_val)

            #validation results
            total_val_loss += loss_val.item()
            preds_val = output_val.argmax(dim=1)
            correct_val += (preds_val == labels_val).sum().item()
            total_number_val += labels_val.size(0)

    avg_loss_train = total_train_loss / len(loader_train)
    accuracy_train = correct_train / total_number_train

    avg_loss_val = total_val_loss / len(loader_val)
    accuracy_val = correct_val / total_number_val

    scheduler.step(avg_loss_val)

    print(f"\nTrain Loss: {avg_loss_train:.4g}, Train Accuracy: {accuracy_train*100:.2f}%")
    print(f"Validation Loss: {avg_loss_val:.4g}, Validation Accuracy: {accuracy_val*100:.2f}%")

    return avg_loss_train, accuracy_train, avg_loss_val, accuracy_val

def save_model(model, num_classes, path):
    """Saves the model state_dict and architecture metadata."""
    torch.save({
        'model_class': 'CNN',
        'architecture': {
            'conv1': {'in_channels': 1, 'out_channels': 16, 'kernel_size': 3, 'padding': 1},
            'conv2': {'in_channels': 1, 'out_channels': 32, 'kernel_size': 3, 'padding': 1},
            'conv3': {'in_channels': 1, 'out_channels': 64, 'kernel_size': 3, 'padding': 1},
            'conv4': {'in_channels': 32, 'out_channels': 128, 'kernel_size': 3, 'padding': 1},
            'conv5': {'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'padding': 1},
            'conv6': {'in_channels': 256, 'out_channels': 512, 'kernel_size': 3, 'padding': 1},
            'conv7': {'in_channels': 512, 'out_channels': 1024, 'kernel_size': 3, 'padding': 1},
            'fc1': {'in_features': 1024 * 2 * 2, 'out_features': 256},
            'fc2': {'in_features': 256, 'out_features': num_classes},
            'activation': 'ReLU',
            'dropout': 'f1 and f2',
            'pooling': 'MaxPool2d(kernel_size=2)',
        },
        'state_dict': model.state_dict()
    }, path)

    print("model saved")

def save_training_log(root, base_name, epoch_list, train_accuracy_list, val_accuracy_list, train_loss_list, val_loss_list):
    """Logs training/validation metrics to CSV."""
    filename = os.path.join(root, f"{base_name}_log.csv")
    epoch_list = [int(i) for i in epoch_list]
    data = np.column_stack((epoch_list, train_accuracy_list, val_accuracy_list, train_loss_list, val_loss_list))
    header = "epoch,train_accuracy,val_accuracy,train_loss,val_loss"
    np.savetxt(filename, data, delimiter=",", header=header, comments='', fmt="%.6f")

    print("log saved")

def model_update(model, path):
    """Loads saved model weights into the given model instance."""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])

#plot functions
def plot_one_image(loader, classes):
    """Displays the first image from a loader using its class label."""
    images, labels = next(iter(loader))
    plot_random_image(images[0], labels[0], classes)

def plot_random_image(image_tensor, label, classes):
    """Displays a single grayscale image with a label."""
    image_np = image_tensor.squeeze().numpy()  # remove channel dim if it's [1, H, W]
    plt.imshow(image_np, cmap='gray')  # specify grayscale colormap
    plt.title(f"Label: {classes[label]}")
    plt.axis('off')
    plt.show()

def plot_comparison(epoch_list, train_results_list, val_results_list, title):
    """Plots training vs validation curves (loss or accuracy)."""

    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel(f'Train {title}', color=color)
    ax1.plot(epoch_list, train_results_list, marker='o', color=color, label=f'Train {title}')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second y-axis
    color = 'tab:red'
    ax2.set_ylabel(f'val {title}', color=color)
    ax2.plot(epoch_list, val_results_list, marker='x', color=color, label=f'Accuracy {title}')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title(f"Train {title} and Validation {title}")
    plt.show()


def main():
    """Main training loop, configuration, and experiment control."""
    #result list initialisation
    epoch_list, train_loss_list, train_accuracy_list, val_loss_list, val_accuracy_list  = [], [], [], [], []

    # === CONFIGURATION ===
    USE_DEFAULT_PATHS = True

    if USE_DEFAULT_PATHS:
        root_train = os.path.normpath(os.path.join("..", "data_preparation", "dataset", "train_data"))
        root_val = os.path.normpath(os.path.join("..", "data_preparation", "dataset", "val_data"))
        root_log = os.path.normpath(os.path.join("..", "Results"))
        checkpoint_root = os.path.normpath(os.path.join("..", "model_trained"))
    else:
        root_train = "path/to/train"
        root_val = "path/to/val"
        root_log = "path/to/logs"
        checkpoint_root = "path/to/checkpoints"

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(torch.cuda.get_device_name(0))
    transform_train = build_transform_train()
    transform_val = build_transform_val()
    num_classes, classes = find_classification(root_train)

    # Data
    loader_train, loader_val = data_loader(root_train, root_val, transform_train, transform_val, subset_size = False)

    # Plot one image
    # Optional: uncomment to show one example image
    # plot_one_image(loader_train, classes)

    # Model
    base_name = "example_model" #base name of the checkpoint saved
    checkpoint_path = os.path.join(checkpoint_root, base_name) + ".pth"

    model = GalaxyCNN(num_classes).to(device)
    if os.path.exists(checkpoint_path):
        model_update(model, checkpoint_path)
        print("Using pretrained model.")
    else:
        print("Training new model from scratch.")
        save_model(model, num_classes, checkpoint_path)

    criterion = build_criterion(root_train, device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.5, patience=3,
                                                     verbose=True)
    # Training
    max_epochs = 30
    try:
        for epoch in range(1, max_epochs + 1):
            train_loss, train_accuracy, val_loss, val_accuracy = train_val(model, loader_train, loader_val,
                                                                                 optimizer, scheduler, criterion, epoch, device)

            train_loss_list.append(train_loss)
            train_accuracy_list.append(train_accuracy)
            val_loss_list.append(val_loss)
            val_accuracy_list.append(val_accuracy)
            epoch_list.append(epoch)

            new_base_name = f"{base_name}_{epoch}"
            new_checkpoint_path = os.path.join(checkpoint_root, new_base_name) + ".pth"
            save_model(model, num_classes, path= new_checkpoint_path)
            

    except:
        print("\nTraining interrupted by user.")

    plot_comparison(epoch_list, train_loss_list, val_loss_list, title = "loss")
    plot_comparison(epoch_list, train_accuracy_list, val_accuracy_list, title = "accuracy")
    save_training_log(root_log, base_name, epoch_list, train_accuracy_list, val_accuracy_list, train_loss_list, val_loss_list)
    return model, train_loss_list, train_accuracy_list, val_loss_list, val_accuracy_list

model, train_loss_list, train_accuracy_list, val_loss_list, val_accuracy_list = main()

#printing the log in case the log was not saved properly
print("train loss to copy")
for i in train_loss_list:
    print(i)
print("train acc to copy")
for i in train_accuracy_list:
    print(i)

print("val loss to copy")
for i in val_loss_list:
    print(i)

print("val acc to copy")
for i in val_accuracy_list:
    print(i)







