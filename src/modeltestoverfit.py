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





train_size = int(0.8 * len(dtd_dataset))
test_size = len(dtd_dataset) - train_size
train_dataset, test_dataset = data_utils.random_split(dtd_dataset, [train_size, test_size])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

print(len(train_loader.dataset))
print(len(test_loader.dataset))

class MobileNet(nn.Module):
    def __init__(self, num_classes=47):
        super(MobileNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            self._make_layer(32, 64, stride=1,groups=32),
            self._make_layer(64, 128, stride=2,groups=64),
            self._make_layer(128, 128, stride=1,groups=128),
            self._make_layer(128, 256, stride=2,groups=128),
            
            self._make_layer(256, 256, stride=1,groups=256),
            self._make_layer(256, 512, stride=2,groups=256),
            
            # Repeat the following block 5 times
            # self._make_layer(512, 512, stride=1,groups=32),
            # self._make_layer(512, 512, stride=1,groups=32),
            # self._make_layer(512, 512, stride=1),
            # self._make_layer(512, 512, stride=1),
            # self._make_layer(512, 512, stride=1),
            
            # self._make_layer(512, 1024, stride=2),
            # self._make_layer(1024, 1024, stride=1),
            
            
            # self._make_layer(512, 256, stride=1,groups=256),
            # self._make_layer(256, 128, stride=1,groups=128),
            # self._make_layer(128, 64, stride=1,groups=64),

            nn.AdaptiveAvgPool2d(1)
           
        )
        self.dropout = nn.Dropout(p=0.2, inplace=True)
        self.fc = nn.Linear(512, num_classes)
        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, stride,groups):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1,groups=groups, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.model(x)
        x= self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Create an instance of MobileNet
model = MobileNet()
# Print the architecture
print(sum(p.numel() for p in model.parameters()))


# Define the optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
# optimizer = optim.SGD(model.parameters(), lr=1e-03, momentum=0.9)
criterion = nn.CrossEntropyLoss()

filename = 'checkpoints\savebestmodel.pt'
checkpoint = torch.load(filename,map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict'])

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
