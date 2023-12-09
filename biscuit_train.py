import os
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn

from torchvision.models import resnet18
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from biscuit_custom_dataset import BiscuitDataset

def train(model, train_loader, val_loader, epochs, optimizer, criterion, device) :
    best_val_acc = 0.0
    train_losses_list = []
    val_losses_list = []

    for epoch in range(epochs) :
        train_loss = 0.0
        total_correct = 0.0
        total_samples = 0.0

        for batch_index, (images, labels) in enumerate(train_loader) :
            image = images.to(device)
            label = labels.to(device)

            optimizer.zero_grad()
            outputs = model(image)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            if batch_index % 10 == 0 :
                print(f"Epoch {epoch+1}/{epochs},"
                      f"Batch {batch_index+1}/{len(train_loader)},"
                      f"Loss {loss.item()}")
                train_loss += loss.item()

        model.eval() # train -> eval
        total_val_loss = 0.0

        for images, labels in val_loader :
            image = images.to(device)
            label = labels.to(device)
            outputs = model(image)
            loss = criterion(outputs, label)
            total_val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == label).sum().item()
            total_samples += label.size(0) # (batch size ch w h)

        val_acc = total_correct / total_samples
        average_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs} "
              f"train_loss {train_loss:.4f},"
              f"val_loss {average_val_loss:.4f}"
              f"val acc {val_acc:.4f}")
        val_losses_list.append(average_val_loss)
        train_losses_list.append(train_loss)

        if val_acc > best_val_acc :
            torch.save(model.state_dict(), 'best.pt')
            best_val_acc = val_acc

        model.train() # model eval() -> model train()

    return train_losses_list, val_losses_list





def main() :
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device {device}")

    # transforms
    train_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(p=.4),
        transforms.RandomVerticalFlip(p=.4),
        transforms.RandomRotation(degrees=5),
        transforms.AutoAugment(),
        transforms.AugMix(),
        transforms.ToTensor()
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    # dataset dataloader
    train_dataset = BiscuitDataset("./dataset/train/",
                                   transform=train_transforms)
    val_dataset = BiscuitDataset("./dataset/val/",
                                   transform=val_transforms)
    train_loader = DataLoader(train_dataset, batch_size=46,
                              shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=46, shuffle=False)

    # model = resnet18
    model = resnet18(weights=True)
    #   (fc): Linear(in_features=512, out_features=1000, bias=True)
    model.fc = nn.Linear(in_features=512, out_features=65)
    model.to(device)

    # loss optim
    criterion = CrossEntropyLoss().to(device)
    optimizer = AdamW(model.parameters(), lr=0.001)

    train_losses, val_losses = train(model, train_loader, val_loader,
                                     20, optimizer, criterion, device)

    plt.plot(train_losses, label="Train loss")
    plt.plot(val_losses, label="Val loss")
    plt.legend()
    plt.show()



if __name__ == "__main__" :
    main()