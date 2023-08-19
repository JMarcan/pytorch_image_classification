import json
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import requests
JSON_CONTENT_TYPE = 'application/json'
PNG_CONTENT_TYPE = 'image/png'

def net():
    NUM_CLASSES = 4
    
    # Load the pretrained model
    model = models.resnet50(pretrained=True) 
    
    # Freeze the parameters of the model to use for feature extraction
    for param in model.parameters():
        param.requires_grad = False
        
    # Configure output layer to classify for our 4 classes
    model.fc = nn.Sequential(
        nn.Linear(2048, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, NUM_CLASSES))
    
    return model

# Based on: https://github.com/aws/amazon-sagemaker-examples/blob/main/frameworks/pytorch/get_started_mnist_deploy.ipynb
def model_fn(model_dir):
    model = net()
    
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    model.eval()
    return model

def input_fn(request_body, request_content_type):
    if content_type == PNG_CONTENT_TYPE: 
        return Image.open(io.BytesIO(request_body))
    
    raise Exception(f"Requested unsupported ContentType in content_type: {content_type} instead of image/png")

def predict_fn(input_object, model):
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    input_object=test_transform(input_object)
    
    with torch.no_grad():
        prediction = model(input_object.unsqueeze(0))
    return prediction