import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output


def compute_accuracy(dataloader, model, device = "cpu"):
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