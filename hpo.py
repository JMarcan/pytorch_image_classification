#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import argparse

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def test(model, test_loader, criterion, device):
    test_loss = 0
    correct = 0
    movel.eval()
    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, label, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )

def train(model, train_loader, loss_criterion, optimizer, device):
    for batch_idx, (data, label) in enumerate(train_loader):
        data = data.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
    
def net():
    num_classes = 4
    
    model = models.resnet50(pretrained=True) # load the pretrained model
    
    # freeze the parameters of the model to use for feature extraction
    for param in model.parameters():
        param.requires_grad = False
        
    num_inputs = model.fc.in_features
    # configure output layer to classify for our 4 classes
    model.fc = nn.Linear(num_inputs, num_classes)
    
    return model

def create_data_loaders(data, batch_size):
    train_data_path = os.path.join(data, "TRAIN/")
    test_data_path = os.path.join(data, "TEST/")
    validation_data_path = os.path.join(data, "TEST_SIMPLE/")
    
    train_data = torchvision.datasets.ImageFolder(
        root=train_data_path,
        transform=train_transform
    )
    train_data_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
    )
    
    
    test_data = torchvision.datasets.ImageFolder(
        root=test_data_path,
        transform=test_transform
    )
    test_data_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
    )
    
    validation_data = torchvision.datasets.ImageFolder(
        root=validation_data_path,
        transform=test_transform,
    )
    validation_data_loader = torch.utils.data.DataLoader(
        validation_data,
        batch_size=batch_size,
        shuffle=False,
    )
    
    return train_data_loader, test_data_loader, validation_data_loader

def main(args):
    logger.info(f"[ Hyperparameters ] LR: {args.lr} | Batch Size: {args.batch_size} | Epochs: {args.epochs} | Momentum: {args.momentum}")
    logger.info(f"Data Paths: {args.data}")
         
    # Initialize a model by calling the net function
    model=net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Create loss and optimizer
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate, momentum=args.momentum)
         
    train_loader, test_loader, validation_loader = create_data_loaders(args.data,
                                                                       args.batch_size)
    model=train(model, train_loader, loss_criterion, optimizer, device)
    
    test(model, test_loader, criterion, device)
    
    torch.save(model, path)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    
    parser.add_argument("--lr",
                        type=float,
                        default=0.01)
    parser.add_argument("--batch-size",
                        type=int,
                        default=32)
    parser.add_argument("--epochs",
                        type=int,
                        default=10)
    parser.add_argument("--momentum",
                        type=float,
                        default=0.5)
    parser.add_argument("--data", type=str,
                        default="blood-cells/dataset2-master/dataset2-master/images")
    parser.add_argument("--model_dir",
                        type=str,
                        default="s3://image-classification-blood-cells/trained_model")
    parser.add_argument("--output_dir",
                        type=str,
                        default="s3://image-classification-blood-cells/trained_model")
    
    args=parser.parse_args()
    print(args)
    
    main(args)
