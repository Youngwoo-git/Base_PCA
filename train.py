import os
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torchvision.models.vgg import vgg11, vgg13, vgg16, vgg19
import argparse
from utils.train_utils import create_dataset, trainer

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
    save_dir = os.path.join(args.save_dir, date_time+"_"+args.model+"_"+args.dataset)
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
    