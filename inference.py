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
JPG_CONTENT_TYPE = 'image/jpeg'

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

def model_fn(model_dir):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = net().to(device)
    
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f, map_location=device))
    model.eval()
    return model

def input_fn(request_body, request_content_type):
    if content_type == JPG_CONTENT_TYPE: 
        return Image.open(io.BytesIO(request_body)
    else:
        raise Exception(f"Requested unsupported. Expected: {JPG_CONTENT_TYPE}. Received: {content_type}")

def predict_fn(input_object, model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    input_object=test_transform(input_object)
    input_object=input_object.to_device()
    
    with torch.no_grad():
        prediction = model(input_object.unsqueeze(0))
    return prediction