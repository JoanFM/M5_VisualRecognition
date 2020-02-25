import os
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from models import *


def main():
    # Parameters
    batch_size = 16
    epochs = 200
    input_shape = (256,256,3)
    num_classes = 10
    learning_rate = 1e-3
    data_dir = '/home/mcv/datasets/MIT_split'
    work_dir = '/home/grupo07/week1/work'
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    # Create model
    net = SE_v3(input_shape=input_shape, num_classes=num_classes).cuda()

    # TODO: Prepare data

    # Train model
    print('Using GPU: ', torch.cuda.get_device_name(0))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=10)

    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        for i, data in tqdm(enumerate(trainloader)):
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print('[{}, {}] loss: {:.5f}'.format(epoch + 1, i + 1, running_loss))
            running_loss = 0.0

        # TODO: Compute train loss

        # Compute train accuracy
        correct = 0
        total = 0
        with torch.no_grad():
            for data in trainloader:
                inputs, labels = data
                inputs = inputs.cuda()
                labels = labels.cuda()
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total = += labels.size(0)
                correct = (predicted == labels).sum().item()
        acc = correct / total

        # TODO: Compute validation loss

        # Compute validation accuracy
        correct = 0
        total = 0
        with torch.no_grad():
            for data in validationloader:
                inputs, labels = data
                inputs = inputs.cuda()
                labels = labels.cuda()
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total = += labels.size(0)
                correct = (predicted == labels).sum().item()
        val_acc = correct / total

        print('[{}] loss: {:.5f} acc: {:.5f} val_loss: {:.5f} val_acc: {:.5f}'.format(epoch + 1, loss, acc, val_loss, val_acc))

        # ReduceLROnPlateau
        scheduler.step(val_loss)


if __name__ == '__main__':
    main()
