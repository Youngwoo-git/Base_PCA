import os
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from torchvision.models.vgg import vgg11, vgg13, vgg16, vgg19 
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output
from datetime import datetime
import argparse

def create_dataset(dataset = "cifar10", data_root = "./data", batch_size = 2048, num_workers = 2):

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) ## subtract and then divide by 0.5 for each value in every channel
    
    traindataset, testdataset = None, None
    
    if dataset == "cifar10":
        traindataset = torchvision.datasets.CIFAR10(root=data_root, train=True,
                                                download=True, transform=train_transform)
        testdataset = torchvision.datasets.CIFAR10(root=data_root, train=False,
                                               download=True, transform=test_transform)
    elif dataset == "cifar100":
        traindataset = torchvision.datasets.CIFAR100(root=data_root, train=True,
                                                download=True, transform=train_transform)
        testdataset = torchvision.datasets.CIFAR100(root=data_root, train=False,
                                               download=True, transform=test_transform)
    
    trainloader = torch.utils.data.DataLoader(traindataset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers) #: It shuffles the training data at the beginning of each epoch, which helps in randomizing the order in which the samples are fed to the model during training. This improves the convergence of the model.

    testloader = torch.utils.data.DataLoader(testdataset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)
    
    return traindataset, testdataset, trainloader, testloader


def train(model, dataloader, loss_fn, optimizer, device, ldl, lts): ## ldl = length dataloader, lts = length dataset
    model.train()  ## puts the model on training mode such as enabling gradient computations
    total_loss = 0 ## over current epoch

    total_correct = 0 ## extra, calculate the accuracy on training set during epoch
    for batch in tqdm(dataloader):
        inputs, labels = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad() ## deletes stored gradients
        outputs = model(inputs)

        loss = loss_fn(outputs, labels)

        loss.backward()     ## computes gradients
        optimizer.step()    ## updates parameters

        total_loss += loss.item()

        predictions = outputs.argmax(dim=1)

        correct = (predictions == labels).sum().item()
        total_correct += correct

    return total_loss / ldl, total_correct / lts


def compute_accuracy(dataloader, model, device):
    model.eval()  # switch to evaluation mode

    total_correct = 0
    total_count = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch[0].to(device), batch[1].to(device)
            outputs = model(inputs)
            predictions = outputs.argmax(dim=1)
            correct = (predictions == labels).sum().item()
            total_correct += correct
            total_count += len(labels)

    accuracy = total_correct / total_count

    return accuracy


def trainer(model, trainloader, testloader, loss_fn, optimizer, scheduler, device, ldl, lts, num_epochs, patience, save_dir):
    ## the following lists will be used to plot the loss and accuracy curves by keeping track of the values over epochs
    
    plot_save_dir = os.path.join(save_dir, "plot.png")
    model_save_dir = os.path.join(save_dir, "best_model.pt")
    
    best_epoch = 0
    bestScore = 0
    
    train_losses = []
    test_accs = []
    train_accs = []
    lr = []
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, trainloader, loss_fn, optimizer, device, ldl, lts)
        CurrentScore = compute_accuracy(testloader, model, device)

        lr.append(optimizer.param_groups[0]['lr'])
        train_losses.append(train_loss)
        test_accs.append(CurrentScore)
        train_accs.append(train_acc)
        clear_output(wait=True) # wait for all plots to be shown, then erase them and display the updated ones

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, lr: {lr[-1]:.7f}')
        print(f'Test Acc: {CurrentScore:.4f}, Train Acc: {train_acc:.4f}')
        scheduler.step()

        # initialize 3 subplots to plot the loss curve, learning rate curve and accuracy curve
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].plot(train_losses)
        axs[0].set_title('Training Loss')
        axs[1].plot(lr)
        axs[1].set_title('Learning Rate')
        axs[2].plot(test_accs, label="test")
        axs[2].plot(train_accs, label="train")
        axs[2].set_title('Train/Test Accuracy')
        axs[2].legend()
        # plt.show(block=False)
        plt.savefig(plot_save_dir, bbox_inches='tight')
        print("Patience REM:", epoch - best_epoch)
        print(f'Best Test Accuracy Achieved Till now: {max(test_accs):.4f}')
        if CurrentScore > bestScore:
            bestScore = CurrentScore
            best_epoch = epoch
            torch.save(model.state_dict(), model_save_dir)

        elif epoch - best_epoch >= patience:
            print(f'Validation loss did not improve for {patience} epochs. Training stopped.')
            break

    # Print the best test accuracy achieved
    print(f'Best Test Accuracy: {max(test_accs):.4f}')

def run(args):
    model = None
    if args.model == "vgg11":
        model = vgg11(pretrained = args.pretrained)
    elif args.model == "vgg13":
        model = vgg13(pretrained = args.pretrained)
    elif args.model == "vgg16":
        model = vgg16(pretrained = args.pretrained)
    elif args.model == "vgg19":
        model = vgg19(pretrained = args.pretrained)

    # model= nn.DataParallel(model) ## incase you want to use the x2 gpu option you'd need to enable this
    
    if(torch.cuda.is_available()):
        device = torch.device('cuda')
    elif(torch.backends.mps.is_available()): ## for the macbook users :)
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
        
    model.to(device)
    
    trainset, testdataset, trainloader, testloader = create_dataset(dataset = args.dataset,
                                                                     data_root = args.data_dir,
                                                                     batch_size = args.batch_size,
                                                                     num_workers = args.num_workers)
    ldl = len(trainloader)
    lts = len(trainset)

   
    patience = args.patience          # if model accuracy on validation dataset didn't improve for 25 epochs it will stop and save highest scoring model
    num_epochs = args.epoch

    loss_fn = nn.CrossEntropyLoss()
    

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1) #NEW , 92
    
    now = datetime.now() # current date and time
    date_time = now.strftime("%m_%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, date_time+"_"+args.model)
    os.makedirs(save_dir, exist_ok=True)
    
    trainer(model, trainloader, testloader, loss_fn, optimizer, scheduler, device, ldl, lts, num_epochs, patience, save_dir)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', action='store_true',
            help='if start from pretrained')
    parser.add_argument('-n', '--num-workers', default = 2, type = int,
            help = 'number of workers')
    parser.add_argument('-b', '--batch-size', default = 2048, type = int,
            help = 'training batch size')
    parser.add_argument('-e', '--epoch', default = 150, type = int,
            help = 'max epoch')
    parser.add_argument('-p', '--patience', default = 25, type = int,
            help = 'early stop if best accuracy remains unchanged for the length of patience epochs')
    parser.add_argument('--model', default="vgg11",  choices=['vgg11', 'vgg13', 'vgg16', 'vgg19'],
            help = 'model type')
    parser.add_argument('--data-dir', default = "./data", type = str,
            help = 'root data directory')
    parser.add_argument('--dataset', default = "cifar10", choices=['cifar10', 'cifar100'],
            help = 'dataset type')
    parser.add_argument('--save-dir', default = "./result", type = str,
            help = 'model save directory')
    
    args = parser.parse_args()
    print('==> Options:',args)
    
    run(args)
    # initialize the model
    