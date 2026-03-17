Here it is 👇 (just copy everything inside):

# 🧠 Handwritten Digit Recognition using PyTorch

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)
![Dataset](https://img.shields.io/badge/Dataset-MNIST-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📌 Overview

This project implements a Convolutional Neural Network (CNN) using PyTorch to classify handwritten digits (0–9) from the MNIST dataset.

---

## 📊 Dataset

- Training Images: 60,000  
- Testing Images: 10,000  
- Image Size: 28×28  
- Grayscale images  

---

## 🧠 Model Architecture


Input (1x28x28)
│
Conv2D (10 filters, kernel=5) → ReLU
│
MaxPool (2x2)
│
Conv2D (20 filters, kernel=5) → ReLU
│
MaxPool (2x2)
│
Flatten → 320
│
FC (50) → ReLU
│
FC (10)
│
Softmax


---

## ⚙️ Installation


git clone https://github.com/your-username/handwritten-digit-recognition.git

cd handwritten-digit-recognition
pip install torch torchvision matplotlib numpy


---

## 🧩 Code Explanation

### 1. Import Libraries


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


Used for building, training, and optimizing the neural network.

---

### 2. Load Dataset


from torchvision import datasets
from torchvision.transforms import ToTensor

train_data = datasets.MNIST(root='data', train=True, transform=ToTensor(), download=True)
test_data = datasets.MNIST(root='data', train=False, transform=ToTensor())


Loads MNIST dataset and converts images into tensors.

---

### 3. DataLoader


from torch.utils.data import DataLoader

loaders = {
'train': DataLoader(train_data, batch_size=100, shuffle=True),
'test': DataLoader(test_data, batch_size=100, shuffle=True)
}


Handles batching and shuffling of data.

---

### 4. CNN Model


class CNN(nn.Module):
def init(self):
super(CNN, self).init()

    self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    self.fc1 = nn.Linear(320, 50)
    self.fc2 = nn.Linear(50, 10)

Defines convolutional and fully connected layers.

---

### 5. Forward Pass


def forward(self, x):
x = F.relu(F.max_pool2d(self.conv1(x), 2))
x = F.relu(F.max_pool2d(self.conv2(x), 2))
x = x.view(-1, 320)
x = F.relu(self.fc1(x))
x = self.fc2(x)
return F.log_softmax(x, dim=1)


Processes input through layers to generate predictions.

---

### 6. Device Setup


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)


Uses GPU if available.

---

### 7. Optimizer


optimizer = optim.Adam(model.parameters())


Used to update model weights.

---

### 8. Training


def train(epoch):
model.train()
for data, target in loaders['train']:
data, target = data.to(device), target.to(device)
optimizer.zero_grad()
output = model(data)
loss = F.nll_loss(output, target)
loss.backward()
optimizer.step()


Trains the model using backpropagation.

---

### 9. Testing


def test():
model.eval()
correct = 0
with torch.no_grad():
for data, target in loaders['test']:
data, target = data.to(device), target.to(device)
output = model(data)
pred = output.argmax(dim=1)
correct += pred.eq(target).sum().item()


Evaluates model performance.

---

### 10. Training Loop


for epoch in range(1, 11):
train(epoch)
test()


Runs training for 10 epochs.

---

## 📈 Results

- Accuracy: ~98% on MNIST  
- Fast and efficient CNN model  

---

## 🚀 Future Improvements

- Add Dropout  
- Hyperparameter tuning  
- Use deeper models  
- Deploy as web app  

---

## 👨‍💻 Author

Lakshya Sahu

---

## 📜 License

MIT License

Now you can just:

Copy everything

Paste into README.md

Push to GitHub ✅
