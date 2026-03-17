🧠 Handwritten Digit Recognition using PyTorch










📌 Overview

This project builds a Convolutional Neural Network (CNN) using PyTorch to classify handwritten digits (0–9) from the MNIST dataset.

It demonstrates:

Deep learning fundamentals

CNN architecture design

Model training & evaluation

Prediction and visualization

📊 Dataset

Dataset: MNIST

Training Images: 60,000

Testing Images: 10,000

Image Size: 28×28 pixels

Channels: Grayscale (1 channel)

🧠 Model Architecture
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
LogSoftmax
⚙️ Installation
git clone https://github.com/your-username/handwritten-digit-recognition.git
cd handwritten-digit-recognition
pip install torch torchvision matplotlib numpy
🧩 Detailed Code Explanation
1️⃣ Importing Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

Explanation:

torch → Core PyTorch library

nn → Used to build neural network layers

F → Functional API (activations, pooling, loss)

optim → Optimizers like Adam, SGD

2️⃣ Loading Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

train_data = datasets.MNIST(root='data', train=True, transform=ToTensor(), download=True)
test_data = datasets.MNIST(root='data', train=False, transform=ToTensor())

Explanation:

Downloads MNIST automatically

ToTensor() converts images → PyTorch tensors

Pixel values scaled to [0,1]

3️⃣ DataLoader (Batch Processing)
from torch.utils.data import DataLoader

loaders = {
    'train': DataLoader(train_data, batch_size=100, shuffle=True),
    'test': DataLoader(test_data, batch_size=100, shuffle=True)
}

Explanation:

Splits data into batches (size = 100)

shuffle=True ensures randomness → better learning

Improves training efficiency

4️⃣ CNN Model Definition
class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

Explanation:

Conv2d(1,10) → extracts features from image

Second conv layer → deeper features

Fully connected layers → classification

5️⃣ Forward Pass
def forward(self, x):
    x = F.relu(F.max_pool2d(self.conv1(x), 2))
    x = F.relu(F.max_pool2d(self.conv2(x), 2))
    x = x.view(-1, 320)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return F.log_softmax(x, dim=1)

Step-by-step:

Convolution → feature extraction

ReLU → introduces non-linearity

MaxPooling → reduces dimensions

Flatten → converts to vector

Fully connected → classification

LogSoftmax → probability output

6️⃣ Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)

Explanation:

Uses GPU if available → faster training

Otherwise uses CPU

7️⃣ Optimizer
optimizer = optim.Adam(model.parameters())

Explanation:

Adam optimizer adjusts learning rate automatically

Faster convergence compared to SGD

8️⃣ Training Function
def train(epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(loaders['train']):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)

        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

Explanation:

Forward pass → predictions

Compute loss

Backpropagation (loss.backward())

Update weights (optimizer.step())

9️⃣ Testing Function
def test():
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in loaders['test']:
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()

            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

Explanation:

model.eval() → evaluation mode

no_grad() → disables gradient computation

Calculates accuracy

🔟 Training Loop
for epoch in range(1, 11):
    train(epoch)
    test()

Explanation:

Runs training + testing for 10 epochs

Improves model gradually

1️⃣1️⃣ Prediction & Visualization
def predictor(data, model, device):
    model.eval()
    data = data.to(device)
    output = model(data)
    pred = output.argmax(dim=1, keepdim=True)
    return pred.item()
data, target = test_data[0]
prediction = predictor(data.unsqueeze(0), model, device)

Explanation:

Takes single image

Predicts digit

Returns predicted label

📈 Results

Accuracy: ~98% on MNIST

Fast and efficient CNN model

🚀 Future Improvements

Add Dropout (reduce overfitting)

Hyperparameter tuning

Use advanced models (ResNet)

Deploy using Flask / Streamlit

👨‍💻 Author

Lakshya Sahu

📜 License

MIT License
