import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConv(nn.Module):
    def __init__(self, dim_in, dim_out, s_kernel_size, t_kernel_size=1, t_stride=1, t_padding=0, bias=False):
        super(GraphConv, self).__init__()
        self.s_kernel_size = s_kernel_size
        self.conv = nn.Conv2d(
            dim_in,
            s_kernel_size * dim_out,
            kernel_size=(t_kernel_size, 1), 
            padding=(t_padding, 0), 
            stride=(t_stride, 1),
            bias=bias
        )

    def forward(self, x, A):
        assert A.size(0) == self.s_kernel_size
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.s_kernel_size, kc//self.s_kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', x, A)
        
        return x.contiguous(), A
        