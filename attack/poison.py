import torch
import torch.nn.functional as F

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def pgd_attack(model, image, labels, eps=0.031, alpha=2/255, iters=20,device='cpu') :
    image = image.to(device)
    labels = labels.to(device,dtype=torch.long)
    x =image.detach()
    #x = x + torch.zeros_like(x).uniform_(eps, eps)
    for i in range(iters):
        x.requires_grad_()
        with torch.enable_grad():
            outputs = model(x)
            loss = F.cross_entropy(outputs, labels).to(device)
        grad = torch.autograd.grad(loss, [x])[0]
        #print(grad)
        x = x.detach() + alpha * torch.sign(grad.detach())
        x = torch.min(torch.max(x, image-eps), image+eps)
        x = torch.clamp(x, 0, 1)
    return x

