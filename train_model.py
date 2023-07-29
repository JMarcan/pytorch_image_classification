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
    test_loss = 0
    correct = 0
    model.eval()
        
    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            test_loss += loss_criterion(output, label)  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(label.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100.0 * correct / len(test_loader.dataset)

    logger.info(f"Testing loss: {test_loss}")
    logger.info(f"Testing accuracy: {test_accuracy}")

def train(model, train_loader, loss_criterion, optimizer, device, hook):
    hook.set_mode(smd.modes.TRAIN)
        
    for batch_idx, (data, label) in enumerate(train_loader):
        data = data.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_criterion(output, label)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    batch_idx,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
    return model
    
def net():
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
                   nn.Linear(num_features, 128),
                   nn.ReLU(inplace=True),
                   nn.Linear(128, 133))
    return model

def create_data_loaders(data, batch_size):
    train_data_path = os.path.join(data, "TRAIN")
    test_data_path = os.path.join(data, "TEST")
    validation_data_path = os.path.join(data, "TEST_SIMPLE")
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    train_data = torchvision.datasets.ImageFolder(
        root=train_data_path,
        transform=transform
    )
    train_data_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
    )
    
    test_data = torchvision.datasets.ImageFolder(
        root=test_data_path,
        transform=transform
    )
    test_data_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
    )
    
    validation_data = torchvision.datasets.ImageFolder(
        root=validation_data_path,
        transform=transform
    )
    validation_data_loader = torch.utils.data.DataLoader(
        validation_data,
        batch_size=batch_size,
        shuffle=False,
    )
    
    return train_data_loader, test_data_loader, validation_data_loader

def main(args):
    logger.info(f"[ Hyperparameters ] Learning Rate: {args.lr} | Batch Size: {args.batch_size} | Epochs: {args.epochs}")
    logger.info(f"Data Paths: {args.data}")
         
    # Initialize a model by calling the net function
    model=net()
    
    # Load model to GPU if GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Create loss and optimizer
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
         
    train_loader, test_loader, validation_loader = create_data_loaders(args.data, args.batch_size)
    logger.info(f"[ Number of datapoints ] Train data: {len(train_loader.dataset)} | Validation data:{len(validation_loader.dataset)} | Test data: {len(test_loader.dataset)}")
    
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)
    hook.register_loss(loss_criterion)
    logger.info("Training the model")
    tic = time.perf_counter()
    model=train(model, train_loader, loss_criterion, optimizer, device, hook)
    toc = time.perf_counter()
    logger.info(f"Training took {toc - tic:0.4f} seconds")
    
    logger.info("Testing the model")
    test(model, test_loader, loss_criterion, device, hook)
    
    logger.info("Saving the model")
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",
                        type=float,
                        default=0.01)
    parser.add_argument("--batch-size",
                        type=int,
                        default=32)
    parser.add_argument("--epochs",
                        type=int,
                        default=10)
    parser.add_argument("--data", type=str,
                        default=os.environ['SM_CHANNEL_DATA'])
    parser.add_argument("--model_dir",
                        type=str,
                        default=os.environ['SM_MODEL_DIR'])
    parser.add_argument("--output_dir",
                        type=str,
                        default=os.environ['SM_OUTPUT_DATA_DIR'])
    
    args=parser.parse_args()
    
    main(args)
