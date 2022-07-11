import torch
import torch.nn as nn
import torch.nn.functional as F

from .gcn import GraphConv
from data.graph import Graph


###############################################################################
# Helper Functions
###############################################################################
def print_network(net, out_f=None):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    if out_f is not None:
        out_f.write(net.__repr__() + "\n")
        out_f.write('Total number of parameters: %d\n' % num_params)
        out_f.flush()


###############################################################################
# ST-GCN
###############################################################################
class ST_GCN(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride=1, dropout=0, residual=True):
        super(ST_GCN, self).__init__()
        assert len(kernel_size) == 2  
        assert kernel_size[0] % 2 ==1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = GraphConv(dim_in, dim_out, kernel_size[1])
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                dim_out,
                dim_out,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(dim_out),
            nn.Dropout(dropout, inplace=True)
        )

        if not residual:
            self.residual = lambda x: 0

        elif (dim_in == dim_out) and (stride == 1):
            self.residual = lambda x: x
        
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    dim_in,
                    dim_out,
                    kernel_size=1,
                    stride=(stride, 1)
                ),
                nn.BatchNorm2d(dim_out)
            )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A



###############################################################################
# ST-GCN based Recognizer
###############################################################################
class Recognizer(nn.Module):
    def __init__(self, input_nc, num_class, edge_importance_weighting=True):
        super(Recognizer, self).__init__()
        
        # load graph
        self.graph = Graph()
        A = torch.tensor(self.graph.A3, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)  #  pre-defined adjacency graph

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(input_nc * A.size(1))

        self.st_gcn_networks = nn.ModuleList((
            ST_GCN(input_nc, 64, kernel_size, 1, residual=False),
            ST_GCN(64, 64, kernel_size, 1, dropout=0.5),
            ST_GCN(64, 64, kernel_size, 1, dropout=0.5),
            ST_GCN(64, 64, kernel_size, 1, dropout=0.5),
            ST_GCN(64, 128, kernel_size, 2, dropout=0.5),
            ST_GCN(128, 128, kernel_size, 1, dropout=0.5),
            ST_GCN(128, 128, kernel_size, 1, dropout=0.5),
            ST_GCN(128, 256, kernel_size, 2, dropout=0.5),
            ST_GCN(256, 256, kernel_size, 1, dropout=0.5),
            ST_GCN(256, 256, kernel_size, 1, dropout=0.5),
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))  # edge_importance: 9 ones matrix with the same size of A
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # fcn for prediction
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

    def forward(self, x):
        # data normalization
        N, C, T, V, M = x.size()  # [N, 3, 32, 21, 2]
        x = x.permute(0, 4, 3, 1, 2).contiguous()  # [N, 1, 21, 3, 32]
        x = x.view(N * M, V * C, T)  # [N * 1, 21 * 3, 32] = [N, 63, 32]
        x = self.data_bn(x)  # 1D BN
        x = x.view(N, M, V, C, T)  # [N, 1, 21, 3, 32]
        x = x.permute(0, 1, 3, 4, 2).contiguous()  # [N, 1, 3, 32, 21]
        x = x.view(N * M, C, T, V)  # [N, 3, 32, 21]: GCN input shape

        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)  # apply st-gcn block one by one
        
        # [N, 256, 32//4, 21]

        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])  #  [N, C, T, V] -> x.size()[2:] = [T, V] global pooled
        x = x.view(N, M, -1, 1, 1).mean(dim=1) # [N, 256, 1, 1]

        # prediction
        x = self.fcn(x) # [N, 8, 1, 1]
        x = x.view(x.size(0), -1) # [N, 8]

        return x

    def extract_feature(self, x):
        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # [N, 256, 32//4, 21]

        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])  #  [N, C, T, V] -> x.size()[2:] = [T, V] global pooled
        feature = x.view(x.size()[0], -1)

        # prediction
        x = x.view(N, M, -1, 1, 1).mean(dim=1) # [N, 256, 1, 1]
        x = self.fcn(x) # [N, 8, 1, 1]
        out = x.view(x.size(0), -1) # [N, 8]

        return out, feature
