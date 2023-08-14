import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
import torch
from torchvision.transforms import RandomRotation
import matplotlib.pyplot as plt
import numpy as np


def change(targets):
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


epoch_num = 40
batch_size = 64
Train_Loss_list = []
Test_Loss_list = []
Train_Accuracy_list = []
Test_Accuracy_list = []
cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

transf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    #transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
train_set = CIFAR100(root="./data", train=True, transform=transf, download=True)
train_set.targets = change(train_set.targets)
test_set = CIFAR100(root="./data", train=False, transform=transf, download=True)
test_set.targets = change(test_set.targets)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)


class VGG(nn.Module):

    def __init__(self, features, num_class=20):
        super().__init__()
        self.features = features

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_class)
        )

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)

        return output


def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]
        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)


def vgg(num):
    if num == 11:
        return VGG(make_layers(cfg['A'], batch_norm=True))
    elif num == 13:
        return VGG(make_layers(cfg['B'], batch_norm=True))
    elif num == 16:
        return VGG(make_layers(cfg['D'], batch_norm=True))
    else:
        return VGG(make_layers(cfg['E'], batch_norm=True))


def train(net, train_loader, test_loader, device, l_r=0.01, num_epochs=25):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=l_r)
    net = net.to(device)
    for epoch in range(num_epochs):
        train_loss = 0
        test_loss = 0
        test_corrects = 0
        test_corrects2 = 0
        train_corrects = 0
        Test_total_num = 0
        net.train()
        for step, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            output = net(imgs)
            loss = criterion(output, labels.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pre_lab = torch.argmax(output, 1)
            train_batch_corrects = (torch.sum(pre_lab == labels.data).double() / imgs.size(0))
            train_corrects += train_batch_corrects.item()
            if step % 100 == 0:
                print("train {} {}/{} loss: {} acc: {}".format(epoch, step, len(train_loader), loss.item(),
                                                               train_batch_corrects.item()))

        net.eval()
        for step, (imgs, labels) in enumerate(test_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            output = net(imgs)
            loss = criterion(output, labels.long())
            test_loss += loss.item()
            pre_lab = torch.argmax(output, 1)
            test_batch_corrects = (torch.sum(pre_lab == labels.data).double() / imgs.size(0))
            test_corrects += test_batch_corrects.item()
            test_corrects2 += torch.eq(pre_lab, labels).float().sum().item()
            Test_total_num += imgs.size(dim=0)
            test_acc = test_corrects2 / Test_total_num
        print("epoch: ", epoch, "Test Loss: ", loss.item())
        print("Test_Accuracy: ", test_acc)

        print('Epoch: {}/{}'.format(epoch, num_epochs - 1))
        print('{} Train Loss:{:.4f}'.format(epoch, train_loss / len(train_loader)))
        print('{} Test Acc:{:.4f}'.format(epoch, test_corrects / len(test_loader)))
        Train_Loss_list.append(train_loss / (len(train_loader)))
        Test_Loss_list.append(test_loss / (len(test_loader)))
        Train_Accuracy_list.append(100 * train_corrects / (len(train_loader)))
        Test_Accuracy_list.append(100 * test_corrects / (len(test_loader)))
        torch.save(net.state_dict(), './VGG16.pth')


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = vgg(11)
    train(net, train_loader,
          test_loader, device,
          l_r=0.00003,
          num_epochs=epoch_num)
    x1 = range(0, epoch_num)
    x2 = range(0, epoch_num)
    x3 = range(0, epoch_num)
    x4 = range(0, epoch_num)
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

