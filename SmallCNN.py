import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

path = '/Users/rkomati/pneumoniadetection/PN_DS'

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3,16,kernel_size=3, stride = 1, padding = 1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride = 2)
        self.conv2 = nn.Conv2d(16,32,kernel_size=3, stride = 1, padding = 1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride = 2)
        self.conv3 = nn.Conv2d(16,32,kernel_size=3, stride = 1, padding = 1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1 = nn.Linear(32 * 56 * 56, 64)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(64,2)
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1,32 * 56 * 56)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x
    
#transform image
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.CenterCrop(224),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

#load dataset
dataset = ImageFolder(root=path, transform=transform)

#split the dataset into training and valisation 
train_size = (int)(len(dataset) * 0.8)
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

batch_size = 4

#data loaders
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size = batch_size)

model = CNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.5)

#training loop
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range (num_epochs):
    model.train()
    training_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        training_loss += loss.item() * images.size(0)
    
    model.eval()
    val_loss = 0.0
    val_corrects = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            #print("images: {}, labels: {}, output: {}".format(images.size(), labels.size(), output.size()))
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs,1)
            val_corrects += torch.sum(preds == labels.data)
        
    training_loss = training_loss/len(train_loader.dataset)
    val_loss = val_loss / len(val_loader.dataset)
    val_accuracy = val_corrects.double()/len(val_loader.dataset)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {training_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")