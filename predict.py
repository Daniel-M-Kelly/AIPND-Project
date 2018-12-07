import argparse
import time
import os
import sys

import json
from collections import OrderedDict

import numpy as np

import torch
from torch import nn
from torchvision import models

from PIL import Image


def main():
    #Begin tracking time
    start_time = time.time()
    
    #Retrieve Command Line Arguments
    in_arg = get_input_args()
    
    #Check to see if GPU selected, set device variable accordingly.
    if (in_arg.gpu == True and torch.cuda.is_available()):
        device = 'cuda:0'
    #If GPU is selected but not available, use CPU
    elif (in_arg.gpu == True and not torch.cuda.is_available()):
        print('GPU not available, defaulting to CPU')
        device = 'cpu'
    #Default to use CPU
    else:
        device = 'cpu'
        
    print('Will use device: {} for prediction. \n'.format(device))
    
    #Load the pretrained model from the specified checkpoint.
    model = load_model(in_arg.checkpoint, device)
    
    #Load and process the image to predict
    image = process_image(in_arg.image)
    
    #Load the category names file
    categories =  load_category_names(in_arg.category_names)
    
    #Make predictions
    topk_probs, flower_names = predict(image, model, device, in_arg.top_k, categories)
    
    #Output Results
    output_results(topk_probs,flower_names)
    
    #Calculate total time for program to run.
    total_time = time.time() - start_time
    print('This prediction took {:.0f}m {:.2f}s'.format(total_time // 60, total_time % 60))
    
    return None

def get_input_args():
    
    #Create the parser
    parser = argparse.ArgumentParser()
    
    #Arg for image path
    parser.add_argument('image', metavar='IMAGE FILE', type=str, help='Path to image file')
    
    #Arg for model checkpoint file
    parser.add_argument('checkpoint', metavar='CHECKPOINT FILE', type=str, help='Path to model checkpoint file')
    
    #Arg for JSON Category Name file
    parser.add_argument('--category_names', metavar='CATEGORY MAP FILE', type=str, default='cat_to_name.json', help='Path to category map JSON file. (Default cat_to_name.json)')
    
    #Arg for Device (GPU or CPU)
    parser.add_argument('--gpu', help='Use GPU for prediction', action='store_true')
    
    #Arg for # of top classes to return
    parser.add_argument('--top_k', type=int, default=5,
                        help='# of most probable categories to display.')
    
    return parser.parse_args()

def load_model(checkpoint_path, device):
    #Load the saved checkpoint from the specified path, and map it to the correct device (GPU/CPU)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    #set variables for the model architecture and # of hidden units to rebuild the model classifier correctly.
    arch = checkpoint['arch']
    hidden_units = checkpoint['hidden_units']
    
    #Check which architecture the checkpoint used, and load the same one.
    if arch == 'vgg':
        model = models.vgg19(pretrained=True)
        #Build the classifier with the correct # of input, hidden, and output units.
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(25088, checkpoint['hidden_units'])),
            ('relu', nn.ReLU()),
            ('fc2', nn.Linear(checkpoint['hidden_units'], 102)),
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
        #Print a message if the model of the checkpoint is not vgg or alexnet
        print('Model not supported')
    
    #Freeze the parameters of the pre-trained model.
    for param in model.parameters():
        param.requires_grad = False
    
    #Set the model classifier to the custom classifier created above
    model.classifier = classifier
    #Load the state dict of the checkpoint to the model
    model.load_state_dict(checkpoint['state_dict'])
    #Load the class map from the checkpoint to the model.
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def process_image(image_path):
    #Open the image specified in the path
    image = Image.open(image_path)
    
    #Get the width and height of the image
    width, height = image.size
    
    #Resize the image so the shortest side is 256 pixels, keeping the aspect ratio
    if width < height:
        pct = (256/float(width))
        size = int(float((height)*float(pct)))
        image = image.resize((256,size), Image.ANTIALIAS)
    else:
        pct = (256/float(height))
        size = int(float((width)*float(pct)))
        image = image.resize((size,256), Image.ANTIALIAS)
    
    #Get the new width and height of the image
    width, height = image.size
    
    #Calculate boundaries for center cropping the image
    l = (width - 224)/2
    t = (height - 224)/2
    r = (width + 224)/2
    b = (height + 224)/2
    
    #Crop the image
    image = image.crop((l, t, r, b))
    
    #convert to np.array
    image = np.array(image)/255
    #Normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean)/std
    image = image.transpose(2,0,1)
    
    return image

def load_category_names(category_path):
    #Load the JSON file with the category names, either from default location or CLI arg
    with open(category_path, 'r') as f:
        categories = json.load(f)
    return categories

def predict(image, model, device, topk, categories):
    #Set the model to evaluation mode to improve performance
    model.eval()
    #Make sure the model is using the correct device (CPU/GPU)
    model.to(device)
    
    #Convert the image from a Numpy array to a tensor
    image = torch.from_numpy(image).type(torch.FloatTensor)
    
    #Add a dimension to the beginning of the tensor to account for expected dimension for batch size.
    image = image.unsqueeze(0)
    #Move image to correct device (GPU/CPU)
    image = image.to(device)
    
    #Get the predicted probabilities
    probs = torch.exp(model.forward(image))
    
    #Get the TopK probabilities and labels from the prediction
    topk_probs, topk_labels = probs.topk(topk)
    
    #Move Probs and Lables to CPU so that they can be converted to numpy arrays then lists for displaying and pulling the flower names.
    topk_probs = topk_probs.to('cpu')
    topk_labels = topk_labels.to('cpu')
    topk_probs = topk_probs.detach().numpy().tolist()[0]
    topk_labels = topk_labels.detach().numpy().tolist()[0]
    
    #Create a dictionary for the class to model output index mappings from the model.
    idx_to_class = {val: key for key , val in model.class_to_idx.items()}
    #Get the actual flower names based on the top labels and index_to_class dictionary from the JSON file
    flower_names = [categories[idx_to_class[label]] for label in topk_labels]
    
    return topk_probs, flower_names

def output_results(topk_probs, flower_names):
    #output the flower names and probabilities
    print('{:25} {:13}'.format('Flower Name', 'Probability'))
    print('-' * 37)
    
    print()
    
    [print('{1:25} {0:5.2f}%'.format(topk_probs[idx] * 100, str(flower_names[idx]).title())) for idx in range(len(topk_probs))]
    print()
    
    return None

# Call to main function to run the program
if __name__ == "__main__":
    main()