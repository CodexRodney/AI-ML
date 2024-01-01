import argparse
import torch
import json
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import os
import random
from PIL import Image
import numpy as np

# load checkpoint and rebuild the model
def load_checkpoint(filepath):
	model = models.vgg16(pretrained=True)

	
	checkpoint = torch.load(filepath)
	lr = checkpoint['learning_rate']
	model.classifier = checkpoint['classifier']
	model.load_state_dict(checkpoint['model_state_dict'])
	model.class_to_idx = checkpoint['class_to_idx']
	optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

	input_size = checkpoint['input_size']
	output_size = checkpoint['output_size']
	epoch = checkpoint['epoch']


	return model, optimizer, input_size, output_size, epoch
 
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image)
    
    image = transform(image)
    
    return image

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        image = process_image(image_path)

        # tranform to fit input
        image = image.unsqueeze(0)
        
        # moving image to device
        image = image.to(device)
        image = image.float()
        logps = model(image)
        
        # get probabilities and indices of the topk predictions
        ps = torch.exp(logps)
        top_k_probs, top_k_indices = ps.topk(topk, dim=1)
        
        # convert indices to class labels
        idx_to_class = {val: key for key, val in model.class_to_idx.items()}
        top_classes = [idx_to_class[idx.item()] for idx in top_k_indices[0]]
        
        return top_k_probs.squeeze().cpu().numpy(), top_classes

# Initialize
parser = argparse.ArgumentParser(description="This program predicts flowers' names from their images",
								 usage='''
        needs a saved checkpoint
        python predict.py ( use default image 'flowers/test/1/image_06743.jpg' and root directory for checkpoint)
        python predict.py /path/to/image checkpoint (predict the image in /path/to/image using checkpoint)
        python predict.py --top_k 3 (return top K most likely classes)
        python predict.py --category_names cat_to_name.json (use a mapping of categories to real names)
        python predict.py --gpu (use GPU for inference)''',
								 prog='predict')

## Get path of image
parser.add_argument('path_to_image', action="store", nargs='?', default='flowers/test/1/image_06743.jpg', help="path/to/image")
## Get path of checkpoint
parser.add_argument('path_to_checkpoint', action="store", nargs='?', default='checkpoint.pth', help="path/to/checkpoint")
## set top_k
parser.add_argument('--top_k', action="store", default=1, type=int, help="enter number of guesses", dest="top_k")
## Choose json file:
parser.add_argument('--category_names', action="store", default="cat_to_name.json", help="get json file", dest="category_names")
## Set GPU
parser.add_argument('--gpu', action="store_true", default=False, help="Select GPU", dest="gpu")

## Get the arguments
args = parser.parse_args()

arg_path_to_image =  args.path_to_image
arg_path_to_checkpoint = args.path_to_checkpoint
arg_top_k =  args.top_k
arg_category_names =  args.category_names
# Use GPU if it's selected by user and it is available
if args.gpu and torch.cuda.is_available(): 
	arg_gpu = args.gpu
  
# if GPU is selected but not available use CPU
elif args.gpu:
	arg_gpu = False
	print('GPU is not available, will use CPU...')
	print()

else:
	arg_gpu = args.gpu

# Use GPU if selected by user and is available
device = torch.device("cuda" if arg_gpu else "cpu")
print()
print('Will use {} for prediction...'.format(device))
print()

print()
print("Path of image: {} \nPath of checkpoint: {} \nTopk: {} \nCategory names: {} ".format(arg_path_to_image, arg_path_to_checkpoint, arg_top_k, arg_category_names))
print('GPU: ', arg_gpu)
print()

# Label mapping
print('Mapping from category label to category name...')
print()
with open(arg_category_names, 'r') as f:
    cat_to_name = json.load(f, strict=False)

# Loading model
print('Loading model........................ ')
print()

my_model, my_optimizer, input_size, output_size, epoch  = load_checkpoint(arg_path_to_checkpoint)

my_model.eval()

idx_to_class = {v:k for k, v in my_model.class_to_idx.items()}

print(arg_path_to_image)
probs, classes = predict('{}'.format(arg_path_to_image), my_model, arg_top_k)

print()
print('This model predicts this flower as: ')
print()
for count in range(arg_top_k):
     print('{} ...........{:.3f} %'.format(cat_to_name[idx_to_class[classes[0, count].item()]], probs[0, count].item()))