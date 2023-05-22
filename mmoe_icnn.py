import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from sklearn.metrics import mutual_info_score

import warnings
warnings.filterwarnings("ignore")

# random.seed(3)
# np.random.seed(3)
# seed = 3
# batch_size = 1024
n_epochs = 10
batch_size_train = 32
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
num_experts=5
random_seed = 1
torch.manual_seed(random_seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def data_preparation():
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data/', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_train, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data/', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_test, shuffle=True)
    return train_loader, test_loader

train_loader, test_loader = data_preparation()
train_loader.require_grad = True
test_loader.require_grad = True

def getTensorDataset(my_x, my_y):
    tensor_x = torch.Tensor(my_x)
    tensor_y = torch.Tensor(my_y)
    return torch.utils.data.TensorDataset(tensor_x, tensor_y)


class Expert(nn.Module):
    def __init__(self, experts_out):
        super(Expert, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, kernel_size=3)
        self.conv2 = nn.Conv2d(5, 10, kernel_size=2)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(360, 50)
        self.fc2 = nn.Linear(50, experts_out)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        feature = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # feature.requires_grad=True
        x = feature.view(-1, 360)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x, feature

class Tower(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Tower, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(0.4)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        # out = self.dropout(out)
        out = self.fc2(out)
        # out = self.sigmoid(out)
        return out


def cam(a,b, grad_tensor, grad_tensor2):
    a = a*grad_tensor
    b = b*grad_tensor2

    a = torch.mean(a, dim=1)
    b = torch.mean(b, dim=1)

    a = a * a
    b = b * b

    aa = (a - b) * (a - b) / ((a + 0.00001) * (a + 0.00001) + (b + 0.00001) * (b + 0.00001))
    aa = torch.mean(aa, dim=1)
    aa = torch.mean(aa, dim=1)
    aa = torch.mean(aa, dim=0)
    return aa


def po_score(a,b, grad_tensor, grad_tensor2):
    a = a*grad_tensor
    b = b*grad_tensor2

    a = torch.mean(a, dim=1)
    b = torch.mean(b, dim=1)

    a = (a+0.0001) * (a+0.0001)
    b = (b+0.0001) * (b+0.0001)

    t = torch.Tensor([[1],[2],[3],[4],[5],[6]])
    row_a = torch.mean(a, dim=1)
    row_a = torch.sum(torch.mm(row_a,t),dim=1)/torch.sum(row_a,dim=1)
    col_a = torch.mean(a, dim=2)
    col_a = torch.sum(torch.mm(col_a, t),dim=1) / torch.sum(col_a,dim=1)
    # print(torch.sum(row_a,dim=1))

    row_b = torch.mean(b, dim=1)
    row_b = torch.sum(torch.mm(row_b, t),dim=1) / torch.sum(row_b,dim=1)
    col_b = torch.mean(b, dim=2)
    col_b = torch.sum(torch.mm(col_b, t),dim=1) / torch.sum(col_b,dim=1)

    aa = ((row_a-row_b)*(row_a-row_b)+(col_a-col_b)*(col_a-col_b))
    # print(aa)
    aa = torch.mean((aa - torch.mean(aa))*(aa - torch.mean(aa)))
    # print(aa)
    return aa


class MMOE(nn.Module):
    def __init__(self, num_experts, experts_out, experts_hidden, towers_hidden, tasks):
        super(MMOE, self).__init__()
        self.num_experts = num_experts
        self.experts_out = experts_out
        self.experts_hidden = experts_hidden
        self.towers_hidden = towers_hidden
        self.tasks = tasks

        self.softmax = nn.Softmax(dim=1)

        self.experts = nn.ModuleList([Expert(self.experts_out,) for i in range(self.num_experts)])
        self.w_gates = nn.ParameterList([nn.Parameter(torch.randn(28, num_experts), requires_grad=True) for i in range(self.tasks)])
        self.towers = nn.ModuleList([Tower(self.experts_out, 1, self.towers_hidden) for i in range(self.tasks)])

    def forward(self, x, grad_tensor):
        experts_o = [e(x)[0] for e in self.experts]
        experts_o_tensor = torch.stack(experts_o)

        feature_list = [e(x)[1] for e in self.experts]
        # print(feature_list[0].shape)

        mutual_info = [cam(feature_list[i], feature_list[j], grad_tensor[i], grad_tensor[j])
                       for i in range(len(feature_list)) for j in range(len(feature_list))]
        mutual_info = torch.stack(mutual_info)
        mutual_info = torch.mean(mutual_info)

        position_score = [po_score(feature_list[i], feature_list[j], grad_tensor[i], grad_tensor[j])
                          for i in range(len(feature_list)) for j in range(len(feature_list))]
        position_score = torch.stack(position_score)
        position_score = torch.mean(position_score)
        # print(mutual_info)

        # print(aa)

        # print(experts_o)
        # experts_mis = [torch.mean(x, dim=0) for x in experts_o]
        # print(experts_mis[0].shape)
        # print(mutual_info_score(experts_mis[0], experts_mis[1]))
        # mutual_info = mutual_info_score(experts_mis[0], experts_mis[1])

        # experts_mis = [i-torch.min(i)/torch.max(i)-torch.min(i) for i in experts_o]
        # experts_mis = experts_o
        # mutual_info = [F.mse_loss( i, j) for i in experts_mis for j in experts_mis ]
        # mutual_info=torch.mean(torch.stack(mutual_info))

        # print(x.shape)
        s = torch.mean(x, dim=1)
        # print(s.shape)
        s = torch.mean(s, dim=1)

        gates_o = [self.softmax(s @ g) for g in self.w_gates]


        tower_input = [g.t().unsqueeze(2).expand(-1, -1, self.experts_out) * experts_o_tensor for g in gates_o]
        tower_input = [torch.sum(ti, dim=0) for ti in tower_input]
        # tower_input = experts_o

        final_output = [t(ti) for t, ti in zip(self.towers, tower_input)]
        final_output = torch.stack(final_output)
        final_output = torch.mean(final_output, dim=2)
        final_output = final_output.t()
        final_output = F.log_softmax(final_output)
        # print(final_output.shape)
        # print(final_output)
        return final_output, mutual_info, position_score, feature_list

def r_gard(feature_list):
    for f in feature_list:
        f.retain_grad()
    return




network= MMOE(num_experts=num_experts, experts_out=20, experts_hidden=32, towers_hidden=3, tasks=10)
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)


train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

grad_tensor = [torch.ones(batch_size_train, 10, 6, 6) for _ in range(num_experts)]

def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        print(target.shape)
        optimizer.zero_grad()
        output, mutual_info, position_score, feature_list = network(data, grad_tensor)
        print(output.shape)
        loss = F.nll_loss(output, target)-mutual_info+position_score
        r_gard(feature_list)
        loss.backward()
        optimizer.step()
        # print(output.shape)

        # print(feature_list[5].grad.shape)
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} {:.6f} {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(), mutual_info, position_score))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(network.state_dict(), './model.pth')
            torch.save(optimizer.state_dict(), './optimizer.pth')

            for i in range(len(feature_list)):
                feature = feature_list[i]
                feature = torch.mean(feature, dim=3)
                feature = torch.mean(feature, dim=2)
                feature = feature.expand(6, 6, batch_size_train, 10)
                feature = feature.permute(2, 3, 0, 1)
                feature = (feature - torch.min(feature)) / (torch.max(feature) - torch.min(feature) + 0.0001)
                # print(feature.shape)
                # print(feature)
                p = feature.detach()
                # print(p.shape)
                grad_tensor[i] = p







grad_tensor_test = [torch.ones(batch_size_test, 10, 6, 6) for _ in range(num_experts)]

def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data,target in test_loader:
            output, mutual_info, position_score, feature_list  = network(data, grad_tensor_test)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


test()
for epoch in range(1, n_epochs + 1):
    train(epoch)
    test()












