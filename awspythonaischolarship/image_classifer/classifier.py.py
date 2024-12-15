#!/usr/bin/env python
# coding: utf-8
# Programmed By Benjamin Miller
# # Flower Species Classifier
# 
# This script demonstrates training an image classifier for identifying flower species. The workflow involves:
# - Loading and preprocessing a flower dataset.
# - Training a classifier using a pre-trained deep learning model.
# - Evaluating and saving the trained model.
# - Using the trained model for predictions on new images.

# ## Step 1: Import necessary libraries
# Import essential libraries for data handling, image processing, and neural network implementation.
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
from torch import nn, optim
from PIL import Image
from collections import OrderedDict
from torchvision import datasets, transforms, models
import json

# ## Step 2: Load and preprocess the data
# Define paths to training, validation, and test datasets.
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Define data transformations for training, validation, and testing.
# Training data includes data augmentation for better generalization.
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
}

# Load datasets and apply respective transformations.
dirs = {'train': train_dir, 'valid': valid_dir, 'test': test_dir}
image_datasets = {x: datasets.ImageFolder(dirs[x], transform=data_transforms[x])
                  for x in ['train', 'valid', 'test']}

# Define data loaders for efficient data batching and shuffling.
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True)
               for x in ['train', 'valid', 'test']}

# ## Step 3: Load label mapping
# Map integer labels to flower category names for interpretability.
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# ## Step 4: Build and train the model
# Use a pre-trained VGG13 model and replace its classifier with a custom feed-forward network.
model = models.vgg13(pretrained=True)
for param in model.parameters():
    param.requires_grad = False  # Freeze feature extraction layers

# Define a custom classifier and attach it to the pre-trained model.
classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(25088, 4096)),
    ('relu', nn.ReLU()),
    ('dropout1', nn.Dropout(0.2)),
    ('fc2', nn.Linear(4096, 102)),
    ('output', nn.LogSoftmax(dim=1))
]))
model.classifier = classifier

# Define the loss function and optimizer.
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

# Move model to GPU if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Train the model and validate at each epoch.
epochs = 30
for e in range(epochs):
    for phase in ['train', 'valid']:
        if phase == 'train':
            model.train()
        else:
            model.eval()
        running_loss = 0.0
        running_accuracy = 0

        for inputs, labels in dataloaders[phase]:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_accuracy += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(image_datasets[phase])
        epoch_accuracy = running_accuracy.double() / len(image_datasets[phase])
        print(f"Epoch {e+1}/{epochs} - {phase} Loss: {epoch_loss:.4f} Accuracy: {epoch_accuracy:.4f}")

# ## Step 5: Evaluate on test data
# Test the model's performance on unseen test data.
def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

evaluate_model(model, dataloaders['test'])

# ## Step 6: Save the model checkpoint
# Save the trained model and class-to-index mapping.
model.class_to_idx = image_datasets['train'].class_to_idx
torch.save({'state_dict': model.state_dict(), 'class_to_idx': model.class_to_idx}, 'model_checkpoint.pth')

# ## Step 7: Load the model from checkpoint
# Function to load a model for inference.
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.vgg13(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.class_to_idx = checkpoint['class_to_idx']
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, 4096)),
        ('relu', nn.ReLU()),
        ('dropout1', nn.Dropout(0.2)),
        ('fc2', nn.Linear(4096, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    return model

# ## Step 8: Image preprocessing and prediction
# Preprocess an image for the model.
def process_image(image_path):
    img = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)

# Predict the top K classes of an image.
def predict(image_path, model, topk=5):
    model.eval()
    image = process_image(image_path).to(device)
    with torch.no_grad():
        output = model(image)
    probabilities, indices = torch.topk(F.softmax(output, dim=1), topk)
    class_to_idx_inv = {v: k for k, v in model.class_to_idx.items()}
    classes = [class_to_idx_inv[idx] for idx in indices.cpu().numpy()[0]]
    return probabilities.cpu().numpy()[0], classes

# Example prediction
image_path = 'flowers/test/1/image_06743.jpg'
probs, classes = predict(image_path, model)
print(f"Top probabilities: {probs}")
print(f"Corresponding classes: {classes}")
