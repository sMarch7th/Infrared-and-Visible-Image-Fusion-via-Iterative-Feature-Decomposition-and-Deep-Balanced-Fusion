import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import json
import sys

# 定义Siamese网络
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1)) 
        self.fc1 = nn.Linear(512, 128)  
        self.fc2 = nn.Linear(128, 2)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward_one(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.upsample(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        combined = torch.cat([output1, output2], dim=1)
        combined = self.global_avg_pool(combined)  
        combined = combined.view(combined.size(0), -1)
        combined = F.relu(self.fc1(combined))
        output = self.fc2(combined)
        return output


class PatchDataset(Dataset):
    def __init__(self, root_dir, labels_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        with open(labels_file, 'r') as f:
            self.labels = json.load(f)

        self.folder_list = list(self.labels.keys())

    def __len__(self):
        return len(self.folder_list)

    def __getitem__(self, idx):
        folder_name = self.folder_list[idx]
        imgA_path = f'{self.root_dir}/{folder_name}/A.jpg'
        imgB_path = f'{self.root_dir}/{folder_name}/B.jpg'
        imageA = Image.open(imgA_path)
        imageB = Image.open(imgB_path)
        label = self.labels[folder_name]

        if self.transform:
            imageA = self.transform(imageA)
            imageB = self.transform(imageB)

        return imageA, imageB, label


transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
])

root_dir = './data/cnn_judge/train_cnn_judge_vf'
labels_file = './data/cnn_judge/labels/labels_train_vf'
epoch = 120
batch_size = 128
lr = 0.0001
momentum = 0.9
weight_decay = 0.0005


if __name__ == "__main__":
    dataset = PatchDataset(
        root_dir=root_dir, labels_file=labels_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseNetwork().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0005)
    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    min_loss = float('inf')
    weight = None

    for e in range(epoch):
        epoch_loss = 0
        batch_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            image1, image2, labels = data
            image1, image2, labels = image1.to(
                device), image2.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(image1, image2)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01, norm_type=2)
            optimizer.step()
            batch_loss += loss.item()
            epoch_loss += loss.item()
            if i % 10 == 9:
                print(
                    f'[Epoch {e + 1}, Batch {i + 1}] Loss: {batch_loss / 10}')
                batch_loss = 0.0
            sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d] [loss: %f]"
            % (
                e,
                epoch,
                i,
                loss.item()
            )
        )
        if (epoch_loss < min_loss):
            weight = model.state_dict()
            min_loss = epoch_loss
        
    torch.save(weight, './models/cnn_judge_vf.pth')
    print('Finished Training')
