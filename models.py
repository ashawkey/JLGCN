import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def normalize_graph(A):
    deg_inv_sqrt = A.sum(dim=-1).clamp(min=1).pow(-0.5)
    A = deg_inv_sqrt.unsqueeze(-1) * A * deg_inv_sqrt.unsqueeze(-2)
    return A
    
def pairwise_distance(x, y=None):
    # batched pair-wise distance
    # x: [B, Fin ,N]
    if y is None: 
        y = x
    y = y.permute(0, 2, 1) # [B, N, f]
    A = -2 * torch.bmm(y, x) # [B, N, N]
    A += torch.sum(y**2, dim=2, keepdim=True) # [B, N, 1]
    A += torch.sum(x**2, dim=1, keepdim=True) # [B, 1, N]

    return A

class GCNConv(nn.Module):
    def __init__(self, fin, fout):
        super(GCNConv, self).__init__()

        self.fin = fin
        self.fout = fout
        
        self.W = nn.Conv1d(fin, fout, 1)

    def forward(self, x, A):
        # x: [B, f, N]
        
        x = x @ A
        x = self.W(x)

        return x, A

class GCN(nn.Module):
    def __init__(self, args, A):
        super().__init__()
        
        self.args = args
        # renoramlization
        A = A + torch.eye(A.shape[0]).to(A.device)
        self.A = normalize_graph(A).unsqueeze(0).permute(0,2,1)

        self.dp = nn.Dropout(0.5)

        self.conv1 = GCNConv(args["fin"], 16)
        self.conv2 = GCNConv(16, args["num_classes"]) 

    def forward(self, x):
        x = x.unsqueeze(0).permute(0,2,1)

        x, A = self.conv1(x, self.A)
        x = self.dp(F.relu(x))
        x, A = self.conv2(x, self.A) 

        return x, []


class JGCNConv(nn.Module):
    def __init__(self, fin, fout):
        super(JGCNConv, self).__init__()
        self.fin = fin
        self.fout = fout
        
        #self.R = nn.Parameter(torch.eye(fin))
        self.R = nn.Conv1d(fin, 16, 1, bias=False)

        self.W = nn.Conv1d(fin, fout, 1)

    def forward(self, x, old_A):
        # x: [B, f, N]

        #rx = self.R.unsqueeze(0) @ x
        rx = self.R(x)
        A = pairwise_distance(rx)
        A = torch.exp(-A)
        A = normalize_graph(A)
        A = A + old_A
        A = normalize_graph(A)

        x = x @ A.permute(0,2,1)
        x = self.W(x)

        return x, A

class JGCN(nn.Module):
    def __init__(self, args, A):
        super(JGCN, self).__init__()
        
        self.args = args
        self.A = A.unsqueeze(0)

        self.dp1 = nn.Dropout(args["dp_input"])
        self.dp2 = nn.Dropout(args["dp_output"])

        self.conv1 = JGCNConv(args["fin"], 16)
        self.conv2 = JGCNConv(16, args["num_classes"]) 

    def forward(self, x):
        x = x.unsqueeze(0).permute(0,2,1)

        x = self.dp1(x)
        x, A1 = self.conv1(x, self.A)

        x = F.leaky_relu(x, 0.2)

        x = self.dp2(x)
        x, A2 = self.conv2(x, A1)

        return x, [A1, A2]
