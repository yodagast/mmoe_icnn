import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import argparse, configparser

from attack.evasion import fgsm_attack, mifgsm_attack, ifgsm_attack
from attack.poison import pgd_attack
from model.mmoe_model import MMOE,MMOE_GooleNet,MMOE_DenseNet
from model.googlenet import GoogLeNet
from model.densenet import DenseNet121, DenseNet161, DenseNet169, DenseNet201
from model.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152


def train(train_loader, network, device, optimizer, scheduler, criterion):
    network.train()
    res_loss=0.0
    cnt=0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device, dtype=torch.long)
        data = torch.clamp(data, 0, 1)
        optimizer.zero_grad()
        output = network(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        res_loss+=loss.item()
        cnt+=1
    if(scheduler_args=="reducelr"):
        scheduler.step(loss)
    else:
        scheduler.step()
    return float(res_loss/cnt)
def train_with_val(train_loader,val_loader ,network, device, optimizer, scheduler, criterion):
    network.train(True)
    train_loss=0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device, dtype=torch.long)
        data=torch.clamp(data,0,1)
        optimizer.zero_grad()
        output = network(data)
        loss = criterion(output, target)
        train_loss+=loss
        loss.backward()
        optimizer.step()
    val_loss=0.0
    network.train(False)
    with torch.no_grad():
        for i, (data, target) in enumerate(val_loader):
            vinputs, vlabels = data.to(device), target.to(device, dtype=torch.long)
            vinputs = torch.clamp(vinputs, 0, 1)
            voutputs = network(vinputs)
            vloss = criterion(voutputs, vlabels)
            val_loss += vloss
    print("val log loss: {:.4f}".format(val_loss/(i+1)))
    if(scheduler_args=="reducelr"):
        scheduler.step(val_loss/(i+1))
    else:
        scheduler.step()
    return val_loss/(batch_idx+1)

def test(test_loader, network, device, criterion, test_losses):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device, dtype=torch.long)
            data = torch.clamp(data, 0, 1)
            output = network(data)
            test_loss += criterion(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.6f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def test_perturbed(model, device, test_loader, epsilon, attack):
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]
        loss = criterion(output, target)
        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data
        if attack == "fgsm":
            perturbed_data = fgsm_attack(data, epsilon, data_grad)
        elif attack == "ifgsm":
            perturbed_data = ifgsm_attack(data, epsilon, data_grad)
        elif attack == "mifgsm":
            perturbed_data = mifgsm_attack(data, epsilon, data_grad)
        elif attack == "pgd":
            perturbed_data = pgd_attack(model, data, target, epsilon,device=device)
        output = model(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1]
        correct += final_pred.eq(target.data.view_as(final_pred)).sum()
    final_acc = correct.item() / float(len(test_loader.dataset))
    print("{} Epsilon: {}\tTest Accuracy = {} / {} = {}".format(attack, epsilon, correct, len(test_loader.dataset),
                                                                final_acc))
    return final_acc

def prepare_data(data_path,train_with_val='false',train_bth=32,test_bth=32):
    transform_train = transforms.Compose(
        [transforms.Resize((32, 32)),  # resises the image so it can be perfect for our model.
         transforms.RandomHorizontalFlip(),  # FLips the image w.r.t horizontal axis
         transforms.RandomRotation(10),  # Rotates the image to a specified angel
         transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),  # Performs actions like zooms, change shear angles.
         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Set the color params
         transforms.ToTensor(),  # comvert the image to tensor so that it can work with torch
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize all the images
         ])
    transform = transforms.Compose([transforms.Resize((32, 32)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ])
    CIFAR10_trainset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
    CIFAR10_testset = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)
    train_size = int(0.85 * len(CIFAR10_trainset))
    test_size = len(CIFAR10_trainset) - train_size
    val_loader=None
    if (train_with_val == 'true'):
        train_subset, val_subset = torch.utils.data.random_split(
            CIFAR10_trainset, [train_size, test_size], generator=torch.Generator().manual_seed(1))
        train_loader = DataLoader(train_subset, batch_size=train_bth, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=train_bth, shuffle=True)
        test_loader = DataLoader(CIFAR10_testset, batch_size=test_bth, shuffle=True)
        train_loader.require_grad = True
        val_loader.require_grad = True
        test_loader.require_grad = True
    else:
        train_loader = DataLoader(CIFAR10_trainset, batch_size=train_bth, shuffle=True)
        test_loader = DataLoader(CIFAR10_testset, batch_size=test_bth, shuffle=True)
        train_loader.require_grad = True
        test_loader.require_grad = True
    return train_loader,val_loader,test_loader

def prepare_model():
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    config = configparser.ConfigParser()
    parser.add_argument('--conf', '-c', type=str,
                        help='config file needed', default='./exp/mmoe_resnet18.cfg')
    argparse_args = parser.parse_args()
    config.read(argparse_args.conf)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_path = config["data"]["path"]
    train_with_val=config["data"]["train_with_val"]
    train_bth=int( config["data"]["train_batch"])
    test_bth =int( config["data"]["test_batch"])
    train_loader,val_loader,test_loader=prepare_data(data_path, train_with_val,train_bth,test_bth)

    if config["model"]["model_name"] == 'resnet18':
        network = ResNet18().to(device)
    elif config["model"]["model_name"] == 'mmoe_resnet18':
        network = MMOE(num_experts=5, experts_out=20, experts_hidden=32, towers_hidden=5, tasks=10).to(device)
    elif config["model"]["model_name"] == 'mmoe_googlenet':
        network = MMOE_GooleNet(num_experts=5, experts_out=20, experts_hidden=32, towers_hidden=5, tasks=10).to(device)
    elif config["model"]["model_name"] == 'resnet50':
        network = ResNet50().to(device)
    elif config["model"]["model_name"] == 'googlenet':
        network = GoogLeNet().to(device)
    elif config["model"]["model_name"] == 'densenet':
        network = DenseNet121().to(device)
    else:
        network = ResNet18().to(device)
    opt_args = config["model_param"]["optimizer"]
    if opt_args == 'sgd':
        optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    elif opt_args == 'nesterov':
        optimizer = optim.SGD(network.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    elif opt_args == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-4)
    else:
        optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9)
    scheduler_args=config["model_param"]["scheduler"]
    if scheduler_args=='reducelr':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3, threshold=0.001, mode='max')
    elif scheduler_args == 'steplr':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1, last_epoch=-1)
    else:
        scheduler = torch.optim.lr_sheduler.ExponentialLR(optimizer, gamma=0.99)

    criterion_args = config["model_param"]["criterion"]
    if criterion_args == 'ce':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    test_losses = []
    n_epochs = int(config["model_param"]["epochs"])
    for epoch in range(1, n_epochs + 1):
        print("starting train {} epoch ".format(epoch))
        if(config["data"]["train_with_val"]=='true'):
            train_with_val(train_loader,val_loader, network, device, optimizer, scheduler, criterion)
        else:
            train(train_loader, network, device, optimizer, scheduler, criterion)
    test(test_loader, network, device, nn.NLLLoss(), test_losses)
    attack_method=config["standard_attack"]["attack_method"].split(",")
    # epsilons =[0,0.007,0.01,0.02,0.03,0.05,0.1,0.2,0.3]
    epsilons = [i / 10 for i in [0, 0.007, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3]]
    for attack in attack_method:
        accuracies = []
        for eps in epsilons:
            acc = test_perturbed(network, device, test_loader, eps, attack)
            accuracies.append(acc)




