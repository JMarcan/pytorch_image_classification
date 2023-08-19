import os
import sys
import argparse
import logging
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import smdebug.pytorch as smd

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def test(model, test_loader, loss_criterion, device, hook):
    hook.set_mode(smd.modes.EVAL)
    model.eval()
    running_loss=0
    running_corrects=0
    
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = loss_criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    total_loss = running_loss // len(test_loader)
    total_acc = running_corrects.double() // len(test_loader)

    logger.info(f"Testing Loss: {total_loss}")
    logger.info(f"Testing Accuracy: {total_acc}")

def train(model, train_loader, validation_loader, loss_criterion, optimizer, epochs, device, hook):
    hook.set_mode(smd.modes.TRAIN)
        
    best_loss=1e6
    loss_counter=0
    image_dataset={'train':train_loader, 'valid':validation_loader}

    for epoch in range(epochs):
        logger.info(f"Epoch: {epoch}")
        for phase in ['train', 'valid']:
            if phase=='train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in image_dataset[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = loss_criterion(outputs, labels)

                if phase=='train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss // len(image_dataset[phase])
            epoch_acc = running_corrects // len(image_dataset[phase])
            
            if phase=='valid':
                if epoch_loss<best_loss:
                    best_loss=epoch_loss
                else:
                    loss_counter+=1


            logger.info('{} loss: {:.4f}, acc: {:.4f}, best loss: {:.4f}'.format(phase,
                                                                                 epoch_loss,
                                                                                 epoch_acc,
                                                                                 best_loss))
        if loss_counter==1:
            break
        if epoch==0:
            break
    return model
    
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

def create_data_loaders(data_path, batch_size):
    train_data_path = os.path.join(data_path, "TRAIN")
    test_data_path = os.path.join(data_path, "TEST")
    validation_data_path = os.path.join(data_path, "TEST_SIMPLE")
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        ])
    
    train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=train_transform)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=test_transform)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True,)
    
    validation_data = torchvision.datasets.ImageFolder(root=validation_data_path, transform=test_transform)
    validation_data_loader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, shuffle=True,)
    
    return train_data_loader, test_data_loader, validation_data_loader

def main(args):
    logger.info(f"[ Hyperparameters ] Learning Rate: {args.learning_rate} | Batch Size: {args.batch_size} | Epochs: {args.epochs}")
    logger.info(f"Data Path: {args.data_path}")
         
    # Load data
    train_loader, test_loader, validation_loader = create_data_loaders(args.data_path, args.batch_size)
    logger.info(f"[ Number of datapoints ] Train data: {len(train_loader.dataset)} | Validation data:{len(validation_loader.dataset)} | Test data: {len(test_loader.dataset)}")

    # Initialize a model by calling the net function
    model=net()
    
    # Load model to GPU if GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Create loss and optimizer
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)
    
    # create the SMDebug hook and register to the model and loss function.
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)
    hook.register_loss(loss_criterion)
         
    logger.info("Starting Model Training")
    model=train(model, train_loader, validation_loader, loss_criterion, optimizer, args.epochs, device, hook)
    
    logger.info("Testing Model")
    test(model, test_loader, loss_criterion, device, hook)
    
    logger.info(f"Saving Model to {args.model_dir}")
    torch.save(model.cpu().state_dict(), os.path.join(args.model_dir, "model.pth"))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    
    parser.add_argument("--data_path", type=str)

    parser.add_argument("--learning_rate",
                        type=float,
                        default=0.1)
    parser.add_argument("--batch_size",
                        type=int,
                        default=32)
    parser.add_argument("--epochs",
                        type=int,
                        default=5)
    parser.add_argument("--model_dir",
                        type=str,
                        default=os.environ['SM_MODEL_DIR'])
    
    args=parser.parse_args()
    print(args)
    
    main(args)
