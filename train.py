import argparse
import time
import os
from os import listdir
import copy
import json
from collections import OrderedDict
import numpy as np

import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import transforms, datasets, models

def main():
    
    #Retrieve Command Line Arguments
    in_arg = get_input_args()
    #print(in_arg)
    
    #Create the data_dirs dictionary
    data_dirs = create_dirs(in_arg.data)
    #print(data_dirs)
    
    #Create data transforms dictionary
    data_transforms = create_transforms()
    #print(data_transforms)
    
    #Load datasets, and define dataloaders, dataset sizes
    dataloaders, dataset_sizes, class_to_idx = load_data(data_dirs, data_transforms)
    
    
    #Build model and classifier
    model, criterion, optimizer, scheduler = build_model(in_arg.arch, in_arg.lr, in_arg.units, in_arg.gpu)
    #print(model)
    
    #Train the model
    #Check if GPU selected and available.
    if (in_arg.gpu == True and torch.cuda.is_available()):
        device = 'cuda:0'
    #If GPU is selected but not available, use CPU
    elif (in_arg.gpu == True and not torch.cuda.is_available()):
        print('GPU not available, defaulting to CPU')
        device = 'cpu'
    #Default to use CPU
    else:
        device = 'cpu'
        
    print('Will use device: {} for training. \n'.format(device))
    
    #Train the model
    trained_model = train_model(dataloaders, dataset_sizes, model, criterion, optimizer, scheduler, device, in_arg.epochs)
    
    #Test the model
    test_model(trained_model, dataloaders, dataset_sizes, 'test', device)
    
    #Save the checkpoint file to the specified directory, or default.
    save_checkpoint(in_arg.arch, in_arg.save_dir, trained_model, optimizer, class_to_idx, in_arg.units)
    
    return None

def get_input_args():
    
    #Create the parser
    parser = argparse.ArgumentParser()
    
    #Arg for data path
    parser.add_argument('data', metavar='DIR', type=str, help='Path to data folder')
    
    #Arg for model architecture
    parser.add_argument('--arch', '-a', type=str, default = 'vgg', choices = ('vgg', 'alexnet'),
                        help='Architecture for pretrained model, vgg or alexnet. (Default vgg)')
    
    #Arg for learning rate
    parser.add_argument('--lr', '--learning_rate', type=float, default = 0.001,
                        help='Learning rate. (Default 0.001)')
    
    #Arg for epochs
    parser.add_argument('--epochs', '-e', type=int, default=10,
                       help='Number of epochs. (Default 10)')
    
    #Arg for hidden units
    parser.add_argument('--units', '--hidden_units', type=int, default=4096,
                        help='Number of Hidden Units to use in model. (Default 4096)')
    
    #Arg for Device (GPU or CPU)
    parser.add_argument('--gpu', help='Use GPU for training', action='store_true')
    
    #Arg for path to save checkpoint
    parser.add_argument('--save_dir', '-s', type=str, default='',
                        help='The directory path to save the model checkpoint file. (Default is the current directory.')

    return parser.parse_args()

def create_dirs(data_dir):
    #Create a list of data directories
    dirs = listdir(data_dir)
    
    #Check if directory ends in slash or not & set separator
    if data_dir[-1] == '/':
        separator = ''
    else:
        separator = '/'
    #Create dictionary of directories
    data_dirs = {folder: data_dir + separator + folder for folder in dirs}

    return data_dirs

def create_transforms():
    #Dictionary for transforms
    data_transforms = {
        #Transform for training data, Random rotate, resize, crop, flip, convert to tensor, then normalize.
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ]),
        #Valid and Test data, just reize, centercrop, convert to tensor, and normalize.
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms

def load_data(data_dirs, data_transforms):
    
    #load the image datasets based on the data_dirs dictionary
    image_datasets = {key: datasets.ImageFolder(data_dirs[key], data_transforms[key]) for key in data_dirs}
    
    #Configure the dataloader shuffle the images for better training.
    dataloaders = {key: torch.utils.data.DataLoader(image_datasets[key], batch_size=64, shuffle=True) for key in data_dirs}
    
    #Calculate the size of the datasets for later use.
    dataset_sizes = {key: len(image_datasets[key]) for key in data_dirs}
    
    #Get the index to class map from the training set.
    class_to_idx = image_datasets['train'].class_to_idx

    return dataloaders, dataset_sizes, class_to_idx

def build_model(arch, learning_rate, hidden_units, device):
    #Check with model architecture was specified in CLI arguments. build model accordingly.
    if arch == 'vgg':
        model = models.vgg19(pretrained=True)
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(25088, hidden_units)),
            ('relu', nn.ReLU()),
            ('fc2', nn.Linear(hidden_units, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(9216, hidden_units)),
            ('relu', nn.ReLU()),
            ('fc2', nn.Linear(hidden_units, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
    else:
        print('Model {} is not supported. Choose either vgg or alexnet.'.format(arch))
    #Freeze pre-trained model parameters
    for param in model.parameters():
        param.requires_grad = False
    
    #Configure new classifier, criterion, optimizer, and scheduler. Learning rate based on CLI argument 
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
    
    return model, criterion, optimizer, scheduler

def train_model(dataloaders, dataset_sizes, model, criterion, optimizer, scheduler, device, num_epochs):
    
    #Begin tracking time
    start_time = time.time()
    #Configure model for correct device (GPU or CPU)
    model.to(device)
    #Set best epoch accuracy to 0
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                #If training, put model in training mode
                model.train()  
            else:
                #If validating, put model in evaluation mode for performance
                model.eval()
            
            #Reset the stats for each epoch
            running_loss = 0.0
            running_corrects = 0

            #Load batch data from dataloader
            for inputs, labels in dataloaders[phase]:
                #Move data to correct device (GPU or CPU)
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                #Reset gradients to 0
                optimizer.zero_grad()

                #Enable or disable gradient calcualtions depending on doing training or validation
                with torch.set_grad_enabled(phase == 'train'):
                    #Get Preditictions and loss
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    #Backpropagate loss and step the optimizer if training
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            #Calculate epoch stats for loss and accuracy
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
#During training, the validation loss and accuracy are displayed
            print('{} Loss: {:.4f} Accuracy: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            #If the current epoch is more accurate than any previous ones, save the weights.
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
    
    #Calculate and print training time
    total_time = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(total_time // 60, total_time % 60))
    print('Best Accuracy: {:4f}'.format(best_acc))
    
    #Take the weights from the most accurate epoch and save them to the model.
    model.load_state_dict(best_model_wts)
    
    return model

def test_model(model, dataloaders, dataset_sizes, data_set, device):
   
    #Model to evaluation mode to improve performance
    model.eval()
    
    #Move model to correct device (GPU or CPU)
    model.to(device)
    
    running_corrects = 0
    
    #Load the data for the test set
    for inputs,labels in dataloaders[data_set]:
        #make sure the data is configure for the correct device (GPU/CPU)
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        #Don't need gradient calcs
        with torch.no_grad():
            #Get predictions
            outputs = model.forward(inputs)
            _, preds = torch.max(outputs, 1)
        
        #Calculate & print accuracy
        running_corrects += torch.sum(preds == labels.data)
    total_accuracy = running_corrects.double() / dataset_sizes[data_set]
    print('Accuracy {:.4f}'.format(total_accuracy))
    
    return None   

def save_checkpoint(arch, save_dir, model, optimizer, class_to_idx, hidden_units):
    
    #Save parameters to re-build model later
    model.arch = arch
    model.class_to_idx = class_to_idx
    model.hidden_units = hidden_units
    model.to('cpu')
    
    #Save model to the default directory, or directory specified in CLI arg. Prepend architecture name to filename.
    torch.save({
        'state_dict': model.state_dict(),
        'arch': model.arch, 
        'class_to_idx': model.class_to_idx,
        'hidden_units': model.hidden_units
    }, save_dir + arch + 'checkpoint.pth')
    
    print('Checkpoint file saved to {}'.format(save_dir + arch + 'checkpoint.pth'))
        
    return None

# Call to main function to run the program
if __name__ == "__main__":
    main()