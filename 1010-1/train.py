import os
import torch
from torchvision.models import vgg13
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from custom_dataset_1010 import MyCustomDataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def train(model, train_loader, val_loader, epochs, optimizer, criterion, device) :
    best_val_acc = 0.0
    train_losses_list = []
    val_losses_list = []

    for epoch in range(epochs) :
        train_loss = 0.0
        val_loss = 0.0
        val_acc = 0.0
        for batch_index, (images, labels) in enumerate(train_loader) :
            image = images.to(device)
            label = labels.to(device)

            optimizer.zero_grad()
            outputs = model(image)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            if batch_index % 10 == 9 :
                print(f"Epoch {epoch+1}/{epochs}, Loss {loss.item()}")
                train_loss += loss.item()

        # Eval
        for images, labels in val_loader :
            image = images.to(device)
            label = labels.to(device)
            output = model(image)
            pred = output.argmax(dim=1, keepdim=True)
            val_acc += pred.eq(label.view_as(pred)).sum().item()
            val_loss += criterion(output, label).item()

        val_losses_list.append(val_loss)
        train_losses_list.append(train_loss)

        if val_acc > best_val_acc :
            torch.save(model.state_dict(), 'best.pt')
            best_val_acc = val_acc

    return model, train_losses_list, val_losses_list

def main() :
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    # transforms
    train_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.RandomVerticalFlip(p=.2),
        transforms.RandomRotation(degrees=5),
        transforms.ToTensor()
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    # dataset dataloader
    train_dataset = MyCustomDataset("./dataset/train/",
                                    transform=train_transforms)
    val_dataset = MyCustomDataset("./dataset/val/",
                                    transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

    # model = vgg13
    import torch.nn as nn
    model = vgg13(weights=True)
    model.classifier[1] = nn.LeakyReLU()
    model.classifier[2] = nn.Dropout(p=.3)
    model.classifier[4] = nn.LeakyReLU()
    model.classifier[5] = nn.Dropout(p=.3)
    model.classifier[6] = nn.Linear(in_features=4096, out_features=3)
    model.to(device)

    # loss, optim
    criterion = CrossEntropyLoss().to(device)
    optimizer = SGD(model.parameters(), lr=0.01)




if __name__ == "__main__" :
    main()