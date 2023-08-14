import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import CIFAR100

# 全局变量设置
batch_size = 100
EPOCH = 100
LR = 0.001


def sparse2coarse(targets):
    coarse_labels = np.array([4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                              3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                              6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                              0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                              5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                              16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                              10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                              2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                              16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                              18, 1, 2, 15, 6, 0, 17, 8, 14, 13])
    return coarse_labels[targets]


Train_Loss_list = []
Test_Loss_list = []
Train_Accuracy_list = []
Test_Accuracy_list = []

transf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
train_set = CIFAR100(root="./data", train=True, transform=transf, download=True)
train_set.targets = sparse2coarse(train_set.targets)
test_set = CIFAR100(root="./data", train=False, transform=transf, download=True)
test_set.targets = sparse2coarse(test_set.targets)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)


class Lstm(nn.Module):
    def __init__(self) -> None:
        super(Lstm, self).__init__()
        self.lstm = nn.LSTM(32 * 3, 128, batch_first=True, num_layers=3)
        self.line1 = nn.Linear(128, 512)
        self.line2 = nn.Linear(512, 4096)
        self.line3 = nn.Linear(4096, 20)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        out = self.line1(out[:, -1, :])
        out = self.line2(out)
        out = self.line3(out)
        return out


device = torch.device('cuda')


def train(train_loader, test_loader):
    global test_acc
    model = Lstm().to(device)
    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    for epoch in range(EPOCH):
        train_loss = 0
        test_loss = 0
        test_corrects = 0
        train_corrects = 0
        Train_total_num = 0
        Test_total_num = 0
        model.train()
        for step, (data, label) in enumerate(train_loader):
            data, label = data.to(device), label.to(device)
            data = data.view(-1, 32, 32 * 3)
            out = model(data)
            loss = criteon(out, label.long())
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred = out.argmax(dim=1)
            train_corrects += torch.eq(pred, label).float().sum().item()
            Train_total_num += data.size(dim=0)
            train_acc = train_corrects / Train_total_num
        print("epoch: ", epoch, "Train Loss: ", loss.item())
        print("Train_Accuracy: ", train_acc)
        model.eval()
        for step, (data, label) in enumerate(test_loader):
            data, label = data.to(device), label.to(device)
            data = data.view(-1, 32, 32 * 3)
            out = model(data)
            loss = criteon(out, label.long())
            test_loss += loss.item()
            pred = out.argmax(dim=1)
            test_corrects += torch.eq(pred, label).float().sum().item()
            Test_total_num += data.size(dim=0)
            test_acc = test_corrects / Test_total_num
        print("epoch: ", epoch, "Test Loss: ", loss.item())
        print("Test_Accuracy: ", test_acc)
        Train_Loss_list.append(train_loss / (len(train_loader)))
        Test_Loss_list.append(test_loss / (len(test_loader)))
        Train_Accuracy_list.append(100 * train_acc)
        Test_Accuracy_list.append(100 * test_acc)


if __name__ == '__main__':
    train(train_loader, test_loader)
    x1 = range(0, EPOCH)
    x2 = range(0, EPOCH)
    x3 = range(0, EPOCH)
    x4 = range(0, EPOCH)
    y1 = Test_Accuracy_list
    y2 = Test_Loss_list
    y3 = Train_Loss_list
    y4 = Train_Accuracy_list

    plt.subplot(1, 2, 1)
    plt.plot(x3, y3, '.-')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.plot(x2, y2, '.-')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.subplot(1, 2, 2)
    plt.plot(x4, y4, '.-')
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.plot(x1, y1, '.-')
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.show()
