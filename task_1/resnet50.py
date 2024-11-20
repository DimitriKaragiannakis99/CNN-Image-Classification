import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from sklearn.manifold import TSNE
from tqdm import tqdm  # For loading bar

from time import time 

if __name__ == '__main__':
  
  # Adding logs of time
  program_start_time = time()
  
  # Data Transformation - using ResNet and ImageNet Standards
  transform = transforms.Compose([
      transforms.Resize((224, 224)),  # ResNet requires 224x224 input
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
  ])

  # Loading Dataset
  data_dir = 'Data/Colorectal Cancer/'
  dataset = ImageFolder(root=data_dir, transform=transform)

  # Splitting the dataset into training and testing sets (70/30 split)
  train_size = int(0.7 * len(dataset))
  test_size = len(dataset) - train_size
  train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

  # Creating data loaders
  loader_start_time = time()
  train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
  test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
  print(f'Elapsed time for loading data and getting DataLoaders: {(time() - loader_start_time):.2f} seconds')
  
  # Model Setup
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
  init_model_start_time = time()
  # Adjusting for the deprecation warning
  # model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT) # uses pretrained weights
  # model = models.resnet50(pretrained=False) # does not use pretrained weights
  model = models.resnet50(weights=None) # This is better practice then pretrained=False
  print(f'Elapsed time for init of the model: {(time() - init_model_start_time):.2f} seconds')
  
  # Modifying the final layer to match the number of classes
  num_classes = len(dataset.classes)  # Number of classes in the dataset --> 3 classes
  model.fc = nn.Linear(model.fc.in_features, num_classes)
  model.to(device)

  # Hyperparameters
  num_epochs = 10 
  lr = 0.001
  
  # Loss Function and Optimizer
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=lr)

  # Adding the learning rate scheduler
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # Reduces LR by 0.1 every 5 epochs

  # Training 
  train_acc, test_acc, train_loss, test_loss = [], [], [], []

  training_start_time = time()
  
  for epoch in range(num_epochs):
    epoch_start_time = time()
    # Training phase
    model.train()
    running_loss, running_corrects = 0.0, 0
    # Adding tqdm loading bar
    for inputs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}", leave=False):
      inputs, labels = inputs.to(device), labels.to(device)

      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      # Update training metrics
      _, preds = torch.max(outputs, 1)
      running_loss += loss.item() * inputs.size(0)
      running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects.double() / len(train_dataset)
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc.item())

    # Testing phase
    model.eval()
    running_loss, running_corrects = 0.0, 0
    with torch.no_grad():
      for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Update testing metrics
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(test_dataset)
    epoch_acc = running_corrects.double() / len(test_dataset)
    test_loss.append(epoch_loss)
    test_acc.append(epoch_acc.item())

    # Step the scheduler
    scheduler.step()

    print(f"Epoch {epoch+1}/{num_epochs}, "
          f"Train Loss: {train_loss[-1]:.4f}, Train Acc: {train_acc[-1]:.4f}, "
          f"Test Loss: {test_loss[-1]:.4f}, Test Acc: {test_acc[-1]:.4f}")
    print(f'Elapsed time for epoch {epoch+1}: {(time() - epoch_start_time):.2f} seconds')

  print(f'Elapsed time for training model {(time() - training_start_time):.2f} seconds')
  
  # Feature Extraction for t-SNE
  features, labels_list = [], []
  model.eval()
  with torch.no_grad():
    for inputs, labels in train_loader:
      inputs = inputs.to(device)
      outputs = model(inputs)

      # Extract features before the final layer
      features.append(outputs.cpu())
      labels_list.append(labels)

  features = torch.cat(features).numpy()
  labels_list = torch.cat(labels_list).numpy()

  # Applying t-SNE
  tsne = TSNE(n_components=2, random_state=42)
  features_2d = tsne.fit_transform(features)

  # Plotting t-SNE Results
  plt.figure(figsize=(10, 8))
  sns.scatterplot(
      x=features_2d[:, 0], y=features_2d[:, 1], hue=labels_list, palette='viridis', legend='full'
  )
  plt.title("t-SNE Visualization of ResNet-50 Features")
  plt.xlabel("t-SNE Dimension 1")
  plt.ylabel("t-SNE Dimension 2")
  plt.show()
  
  # # Create the directory if it doesn't exist
  # os.makedirs('task_1', exist_ok=True)

  # # Save the model
  # model_save_path = 'task_1/resnet50_colorectal.pth'
  # torch.save(model.state_dict(), model_save_path)

  # print(f"Model saved to {model_save_path}")

  print(f'Elapsed time for program {(time() - program_start_time):.2f}')