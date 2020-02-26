import os
from tqdm import tqdm

import torch
import torchvision.transforms as T
import torchvision
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from models.SE_sequential import SE_v3


def data_loaders(data_dir, batch_size, train_transforms, val_transforms):
    train_set = torchvision.datasets.ImageFolder(
        root = data_dir + os.sep + 'train',
        transform = train_transforms
    )
    val_set = torchvision.datasets.ImageFolder(
        root= data_dir + os.sep + 'test',
        transform = val_transforms)
    dataloaders = {
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)
    }
    return dataloaders

def transforms():
    train_transforms = T.Compose(
        [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=5),
            T.RandomAffine(degrees=0, translate=(0.2,0.2)),
            T.ToTensor()
        ]
    )
    val_transforms = T.Compose(
        [
            T.ToTensor()
        ]
    )
    return (train_transforms, val_transforms)


def train(model, dataloaders, optimizer, scheduler, epochs):
    loss_train_rec = []
    loss_val_rec = []
    acc_train_rec = []
    acc_val_rec = []

    for epoch in tqdm(range(epochs)):
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for i, (inputs, labels) in tqdm(enumerate(dataloaders[phase])):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    outputs = outputs.squeeze()
                    loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # statistics
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data).item()
                print('[{}, {}] loss: {:.5f}'.format(epoch + 1, i + 1, loss.item()))
            print('[{}, {}]'.format(epoch + 1, i + 1))
            if phase is 'train':
                loss_train_rec.append(running_loss / len(dataloaders[phase].dataset))
                acc_train_rec.append(running_corrects / len(dataloaders[phase].dataset))
                print('train_loss: {:.5f} train_acc: {:.5f}'.format(loss_train_rec[-1], acc_train_rec[-1]))
            else:
                loss_val_rec.append(running_loss / len(dataloaders[phase].dataset))
                acc_val_rec.append(running_corrects / len(dataloaders[phase].dataset))
                print('val_loss: {:.5f} val_acc: {:.5f}'.format(loss_val_rec[-1], acc_val_rec[-1]))
        scheduler.step(epoch)
        return (loss_train_rec, loss_val_rec, acc_train_rec, acc_val_rec)


def plot(loss_train_rec, loss_val_rec, acc_train_rec, acc_val_rec, num_epochs):
    # Plot results
    # ACCURACY
    epoch_axis = list(range(num_epochs))
    plt.plot(epoch_axis,acc_train_rec)
    plt.plot(epoch_axis,acc_val_rec)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(work_dir+os.sep+'accuracy.jpg')
    plt.close()
    # LOSS
    plt.plot(epoch_axis,loss_train_rec)
    plt.plot(epoch_axis,loss_val_rec)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(work_dir+os.sep+'loss.jpg')
    plt.close()


def main():
    # Parameters
    batch_size = 16
    epochs = 200
    input_shape = (256,256,3)
    num_classes = 10
    learning_rate = 2e-3
    data_dir = '/home/mcv/datasets/MIT_split'
    work_dir = '/home/grupo07/week1/work'
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    # Get GPU device
    if torch.cuda.is_available(): 
        device = torch.device('cuda:0')
        print('Using GPU with ['+str(torch.cuda.device_count())+'] GPUs')
    else:
        print('Using CPU (##NOT RECOMMENDED!)')
        device = torch.device('cpu')

    # Create Model
    model = SE_v3().to(device)

    # Prepare data
    (train_transforms, val_transforms) = transforms()
    print('Transforms defined.')
    """
    translate = tuple of maximum absolute fraction for horizontal and vertical translations. 
    For example translate=(a, b), then horizontal shift is randomly sampled in the range
    -img_width * a < dx < img_width * a and vertical shift is randomly sampled in the range
    -img_height * b < dy < img_height * b. Will not translate by default.
    """
    dataloaders = data_loaders(data_dir, batch_size, train_transforms, val_transforms)
    print('Data Loaded.')

    # Prepare Optimizers and Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=10)
    print('Optimizers Defined.')

    # Training
    (loss_train_rec, loss_val_rec, acc_train_rec, acc_val_rec) = train(model, dataloaders, optimizer, scheduler, epochs)
    plot(loss_train_rec, loss_val_rec, acc_train_rec, acc_val_rec, epochs)
    
if __name__ == '__main__':
    main()
