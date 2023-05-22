import torch
import torch.nn as nn
import torch.nn.functional as F
from model.googlenet import GoogLeNet
from model.resnet import ResNet18
from model.densenet import DenseNet121
class Tower(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Tower, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        # out = self.sigmoid(out)
        return out


class MMOE(nn.Module):
    def __init__(self, num_experts=5, experts_out=20, experts_hidden=32, towers_hidden=5, tasks=10):
        super(MMOE, self).__init__()
        self.num_experts = num_experts
        self.experts_out = experts_out
        self.experts_hidden = experts_hidden
        self.towers_hidden = towers_hidden
        self.tasks = tasks
        self.softmax = nn.Softmax(dim=1)
        self.experts = nn.ModuleList([ResNet18(num_classes=self.experts_out) for i in range(self.num_experts)])
        self.w_gates = nn.ParameterList(
            [nn.Parameter(torch.randn(32, num_experts), requires_grad=True) for i in range(self.tasks)])
        self.towers = nn.ModuleList([Tower(self.experts_out, 1, self.towers_hidden) for i in range(self.tasks)])

    def forward(self, x):
        experts_o = [e(x) for e in self.experts]
        experts_o_tensor = torch.stack(experts_o)
        x1 = torch.mean(x, dim=1)
        x1 = torch.mean(x1, dim=1)
        gates_o = [self.softmax(x1 @ g) for g in self.w_gates]
        tower_input = [g.t().unsqueeze(2).expand(-1, -1, self.experts_out) * experts_o_tensor for g in gates_o]
        tower_input = [torch.sum(ti, dim=0) for ti in tower_input]
        final_output = [t(ti) for t, ti in zip(self.towers, tower_input)]
        final_output = torch.stack(final_output)
        final_output = torch.mean(final_output, dim=2).t()
        final_output = F.log_softmax(final_output)
        return final_output

class MMOE_DenseNet(MMOE):
    def __init__(self, num_experts=5, experts_out=20, experts_hidden=32, towers_hidden=5, tasks=10):
        super(MMOE_DenseNet, self).__init__()
        self.num_experts = num_experts
        self.experts_out = experts_out
        self.experts_hidden = experts_hidden
        self.towers_hidden = towers_hidden
        self.tasks = tasks
        self.softmax = nn.Softmax(dim=1)
        self.experts = nn.ModuleList([DenseNet121(out=self.experts_out) for i in range(self.num_experts)])
        self.w_gates = nn.ParameterList(
            [nn.Parameter(torch.randn(32, num_experts), requires_grad=True) for i in range(self.tasks)])
        self.towers = nn.ModuleList([Tower(self.experts_out, 1, self.towers_hidden) for i in range(self.tasks)])

    def forward(self, x):
        experts_o = [e(x) for e in self.experts]
        experts_o_tensor = torch.stack(experts_o)
        x1 = torch.mean(x, dim=1)
        x1 = torch.mean(x1, dim=1)
        gates_o = [self.softmax(x1 @ g) for g in self.w_gates]
        tower_input = [g.t().unsqueeze(2).expand(-1, -1, self.experts_out) * experts_o_tensor for g in gates_o]
        tower_input = [torch.sum(ti, dim=0) for ti in tower_input]
        final_output = [t(ti) for t, ti in zip(self.towers, tower_input)]
        final_output = torch.stack(final_output)
        final_output = torch.mean(final_output, dim=2).t()
        final_output = F.log_softmax(final_output)
        return final_output

class MMOE_GooleNet(MMOE):
    def __init__(self, num_experts, experts_out, experts_hidden, towers_hidden, tasks):
        super(MMOE_GooleNet, self).__init__()
        self.num_experts = num_experts
        self.experts_out = experts_out
        self.experts_hidden = experts_hidden
        self.towers_hidden = towers_hidden
        self.tasks = tasks
        self.softmax = nn.Softmax(dim=1)
        self.experts = nn.ModuleList([GoogLeNet(out=self.experts_out) for i in range(self.num_experts)])
        self.w_gates = nn.ParameterList(
            [nn.Parameter(torch.randn(32, num_experts), requires_grad=True) for i in range(self.tasks)])
        self.towers = nn.ModuleList([Tower(self.experts_out, 1, self.towers_hidden) for i in range(self.tasks)])

    def forward(self, x):
        experts_o = [e(x) for e in self.experts]
        experts_o_tensor = torch.stack(experts_o)
        x1=torch.mean(x,dim=1)
        x1=torch.mean(x1,dim=1)
        gates_o = [self.softmax(x1 @ g) for g in self.w_gates]
        tower_input = [g.t().unsqueeze(2).expand(-1, -1, self.experts_out) * experts_o_tensor for g in gates_o]
        #tower_input = [torch.einsum('abc,cad->bda', g.squeeze(1), experts_o_tensor) for g in gates_o]
        tower_input = [torch.sum(ti, dim=0) for ti in tower_input]
        final_output = [t(ti) for t, ti in zip(self.towers, tower_input)]
        final_output = torch.stack(final_output)
        final_output = torch.mean(final_output, dim=2).t()
        final_output = F.log_softmax(final_output)
        #print(final_output.shape)
        return final_output