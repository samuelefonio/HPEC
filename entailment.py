import torch
import torch.nn as nn

def func_psi(p, K):
    p_norm = torch.norm(p, p=2, dim=1, keepdim=True)
    return torch.asin(K*(1-p_norm**2)/p_norm)

def func_psi_vec(p, K, batch):
    p_norm = torch.norm(p, p=2, dim=1, keepdim=True)
    res = torch.asin(K*(1-p_norm**2)/p_norm)
    res = res.T  
    res = torch.repeat_interleave(res, batch, dim=0)
    return res

def func_angle(p, x):
    dot_prod = torch.sum(x * p[0], dim=1, keepdim=True)
    norm_x = torch.norm(x, p=2, dim=1, keepdim=True)
    norm_p = torch.norm(p, p=2, dim=1, keepdim=True)
    num = dot_prod*(1 + norm_p**2) - norm_p**2*(1 + norm_x**2)
    den = norm_p*(torch.norm(p - x))*torch.sqrt(1 + norm_p**2 * norm_x**2 - 2*dot_prod)
    return torch.acos(num/den)

def func_angle_vec(p, x):
    new_x = x.unsqueeze(1)
    new_p = p.unsqueeze(0)
    sol = new_x*new_p
    dot_prod = sol.sum(dim=-1)
    norm_x = torch.norm(x, p=2, dim=1, keepdim=True)
    norm_p = torch.norm(p, p=2, dim=1, keepdim=True)
    norm_p = norm_p.T
    norm_p = torch.repeat_interleave(norm_p, x.shape[0], dim=0)
    num = dot_prod*(1 + norm_p**2) - norm_p**2*(1 + norm_x**2)
    den = norm_p*(torch.pow((new_x - new_p),2).sum(dim=-1).sum(dim=0)).sqrt()*torch.sqrt(1 + norm_p**2 * norm_x**2 - 2*dot_prod)
    return torch.acos(num/den)

class Entailment_loss(nn.Module):
    def __init__(self, K = 0.1, gamma = 1.):
        super(Entailment_loss, self).__init__()
        self.K = K
        self.gamma = gamma
    
    def forward(self, x, p, labels): 
        angle = func_angle_vec(p, x)
        psi = func_psi_vec(p, self.K, x.shape[0])
        angles = torch.max(torch.zeros_like(angle), angle - psi)
        positive = angles[range(len(labels)), labels]
        mask = torch.ones_like(angles, dtype=torch.bool)
        mask[torch.arange(len(labels)), labels] = False
        negative = angles[mask].reshape(x.shape[0], -1)
        negative = self.gamma - negative
        negative = torch.max(torch.zeros_like(negative), negative)
        negative = torch.sum(negative, dim=1)
        loss = torch.mean(positive + negative)
        return loss

def predict_entailment(x, p):
    angles = func_angle_vec(p, x)
    return torch.argmin(angles, dim=1)


    
