import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
from torchvision.models.mobilenetv2 import InvertedResidual, MobileNet_V2_Weights, MobileNetV2
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


# Define transforms for the data
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

dtd_path = '.data\dtd\dtd\images'

# Load the DTD dataset
# trainset = torchvision.datasets.DTD(root='.data',split='train',download=True, transform=train_transforms)
# trainset = torchvision.datasets.DTD(root='.data', split='train', download=False, transform=train_transforms)
dtd_dataset = torchvision.datasets.ImageFolder(root=dtd_path, transform=train_transforms)

# train_loader = DataLoader(trainset, batch_size=16, shuffle=True, num_workers=2)
# testset = torchvision.datasets.DTD(root='.data',split='test',download=True, transform=test_transforms)
# test_loader = DataLoader(testset, batch_size=16, shuffle=False, num_workers=2)

# Use random_split to split the dataset into train and test sets


train_dataset = torchvision.datasets.DTD(root="data",split='train',download=True, transform=train_transforms)
test_dataset = torchvision.datasets.DTD(root="data",split = 'test',download =True, transform=test_transforms)

print(train_dataset.size())
print(test_dataset.size())



train_size = int(0.8 * len(dtd_dataset))
test_size = len(dtd_dataset) - train_size
train_dataset, test_dataset = data_utils.random_split(dtd_dataset, [train_size, test_size])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)


model = torchvision.models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False

# model = MobileNetV2()
model.classifier = nn.Linear(model.last_channel, 47)

# Define the optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

filename = 'model.pt'
model.load_state_dict(torch.load(filename))

num_epochs = 10
for epoch in range(num_epochs):
    correct = 0
    total = 0
    model.eval()
    # model.to(device)
    running_val_loss = 0.0
    running_val_accuracy = 0.0

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            # inputs,lables = inputs.cuda(),labels.cuda()
            # inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # _, predicted = torch.max(outputs.data, 1)
            # total += labels.size(0)
            # correct += (predicted == labels).sum().item()
             # Update validation statistics
            _, predicted = outputs.max(1)
            correct = predicted.eq(labels).sum().item()
            accuracy = correct / labels.size(0)
            running_val_loss += loss.item()
            running_val_accuracy += accuracy
    
    # Calculate average validation loss and accuracy for the epoch
    val_loss = running_val_loss / len(test_loader)
    val_accuracy = running_val_accuracy / len(test_loader)
    writer.add_scalar("Loss/test", val_loss, epoch)
    writer.add_scalar("accuracy/test", val_accuracy, epoch)
    # print('Epoch %d Validation Accuracy of the network on the test images: %d %%' % (epoch+1,100 * correct / total))
    print(f'Epoch [{epoch+1}/{num_epochs}] Val Loss: {val_loss:.4f} Val Accuracy: {val_accuracy:.4f}')

writer.flush()
print('Finished Training')
writer.close
