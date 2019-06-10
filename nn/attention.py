import torch

from torch import nn
from torch.nn import functional as F

import utils


# # Projection of x onto y
# def proj(x, y):
#   return torch.mm(y, x.t()) * y / torch.mm(y, y.t())
#
#
# # Orthogonalize x wrt list of vectors ys
# def gram_schmidt(x, ys):
#   for y in ys:
#     x = x - proj(x, y)
#   return x
#
#
# # Apply num_itrs steps of the power method to estimate top N singular values.
# def power_iteration(W, u_, update=True, eps=1e-12):
#   # Lists holding singular vectors and values
#   us, vs, svs = [], [], []
#   for i, u in enumerate(u_):
#     # Run one step of the power iteration
#     with torch.no_grad():
#       v = torch.matmul(u, W)
#       # Run Gram-Schmidt to subtract components of all other singular vectors
#       v = F.normalize(gram_schmidt(v, vs), eps=eps)
#       # Add to the list
#       vs += [v]
#       # Update the other singular vector
#       u = torch.matmul(v, W.t())
#       # Run Gram-Schmidt to subtract components of all other singular vectors
#       u = F.normalize(gram_schmidt(u, us), eps=eps)
#       # Add to the list
#       us += [u]
#       if update:
#         u_[i][:] = u
#     # Compute this singular value and add it to the list
#     svs += [torch.squeeze(torch.matmul(torch.matmul(v, W.t()), u.t()))]
#     #svs += [torch.sum(F.linear(u, W.transpose(0, 1)) * v)]
#     return svs, us, vs
#
#
# # Spectral normalization base class
# class SN(object):
#     def __init__(self, num_svs, num_itrs, num_outputs, transpose=False, eps=1e-12):
#         # Number of power iterations per step
#         self.num_itrs = num_itrs
#         # Number of singular values
#         self.num_svs = num_svs
#         # Transposed?
#         self.transpose = transpose
#         # Epsilon value for avoiding divide-by-0
#         self.eps = eps
#         # Register a singular vector for each sv
#         for i in range(self.num_svs):
#             self.register_buffer('u%d' % i, torch.randn(1, num_outputs))
#             self.register_buffer('sv%d' % i, torch.ones(1))
#
#     # Singular vectors (u side)
#     @property
#     def u(self):
#         return [getattr(self, 'u%d' % i) for i in range(self.num_svs)]
#
#     # Singular values;
#     # note that these buffers are just for logging and are not used in training.
#     @property
#     def sv(self):
#         return [getattr(self, 'sv%d' % i) for i in range(self.num_svs)]
#
#     # Compute the spectrally-normalized weight
#     def W_(self):
#         W_mat = self.weight.view(self.weight.size(0), -1)
#         if self.transpose:
#             W_mat = W_mat.t()
#         # Apply num_itrs power iterations
#         for _ in range(self.num_itrs):
#             svs, us, vs = power_iteration(W_mat, self.u, update=self.training,
#                                           eps=self.eps)
#             # Update the svs
#         if self.training:
#             with torch.no_grad():  # Make sure to do this in a no_grad() context or you'll get memory leaks!
#                 for i, sv in enumerate(svs):
#                     self.sv[i][:] = sv
#         return self.weight / svs[0]
#
#
# # 2D Conv layer with spectral norm
# class SNConv2d(nn.Conv2d, SN):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1,
#                  padding=0, dilation=1, groups=1, bias=True,
#                  num_svs=1, num_itrs=1, eps=1e-12):
#         nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride,
#                            padding, dilation, groups, bias)
#         SN.__init__(self, num_svs, num_itrs, out_channels, eps=eps)
#
#     def forward(self, x):
#         return F.conv2d(x, self.W_(), self.bias, self.stride,
#                         self.padding, self.dilation, self.groups)
#
#
# # Linear layer with spectral norm
# class SNLinear(nn.Linear, SN):
#     def __init__(self, in_features, out_features, bias=True,
#                  num_svs=1, num_itrs=1, eps=1e-12):
#         nn.Linear.__init__(self, in_features, out_features, bias)
#         SN.__init__(self, num_svs, num_itrs, out_features, eps=eps)
#
#     def forward(self, x):
#         return F.linear(x, self.W_(), self.bias)
#
#
# # Embedding layer with spectral norm
# # We use num_embeddings as the dim instead of embedding_dim here
# # for convenience sake
# class SNEmbedding(nn.Embedding, SN):
#     def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
#                  max_norm=None, norm_type=2, scale_grad_by_freq=False,
#                  sparse=False, _weight=None,
#                  num_svs=1, num_itrs=1, eps=1e-12):
#         nn.Embedding.__init__(self, num_embeddings, embedding_dim, padding_idx,
#                               max_norm, norm_type, scale_grad_by_freq,
#                               sparse, _weight)
#         SN.__init__(self, num_svs, num_itrs, num_embeddings, eps=eps)
#
#     def forward(self, x):
#         return F.embedding(x, self.W_())


# A non-local block as used in SA-GAN
# Note that the implementation as described in the paper is largely incorrect;
# refer to the released code for the actual implementation.
class AttentionBlock(nn.Module):
    def __init__(self, channels, which_conv=nn.Conv2d, heads=8):
        super(AttentionBlock, self).__init__()
        # Channel multiplier
        self.channels = channels
        self.which_conv = which_conv
        self.heads = heads
        self.theta = self.which_conv(self.channels, self.channels // heads, kernel_size=1, padding=0,
                                     bias=False)
        self.phi = self.which_conv(self.channels, self.channels // heads, kernel_size=1, padding=0,
                                   bias=False)
        self.g = self.which_conv(self.channels, self.channels // 2, kernel_size=1, padding=0,
                                 bias=False)
        self.o = self.which_conv(self.channels // 2, self.channels, kernel_size=1, padding=0,
                                 bias=False)
        # Learnable gain parameter
        self.gamma = nn.Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, inputs, y=None):
        # Apply convs
        theta = self.theta(inputs)
        phi = F.max_pool2d(self.phi(inputs), [2, 2])
        g = F.max_pool2d(self.g(inputs), [2, 2])
        # Perform reshapes
        theta = theta.view(-1, self.channels // self.heads, inputs.shape[2] * inputs.shape[3])
        phi = phi.view(-1, self.channels // self.heads, inputs.shape[2] * inputs.shape[3] // 4)
        g = g.view(-1, self.channels // 2, inputs.shape[2] * inputs.shape[3] // 4)
        # Matmul and softmax to get attention maps
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
        # Attention map times g path
        o = self.o(torch.bmm(g, beta.transpose(1, 2)).view(-1, self.channels // 2, inputs.shape[2],
                                                           inputs.shape[3]))
        outputs = self.gamma * o + inputs
        return outputs


class ConvAttentionNet(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 num_blocks
                 ):
        super().__init__()
        self.initial_layer = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, padding=0)
        self.attention_blocks = nn.ModuleList([
            AttentionBlock(
                channels=hidden_channels,
                which_conv=nn.Conv2d,
                heads=8
            ) for _ in range(num_blocks)
        ])
        # if use_batch_norm:
        self.batch_norm_layers = nn.ModuleList([
            nn.BatchNorm2d(
                num_features=hidden_channels
            )
        ])
        self.final_layer = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, inputs):
        temps = self.initial_layer(inputs)
        for attention, batch_norm in zip(self.attention_blocks, self.batch_norm_layers):
            temps = attention(temps)
            temps = batch_norm(temps)
        outputs = self.final_layer(temps)
        return outputs


def main():
    batch_size, channels, height, width = 100, 12, 64, 64
    inputs = torch.rand(batch_size, channels, height, width)
    net = ConvAttentionNet(
        in_channels=channels,
        out_channels=2 * channels,
        hidden_channels=32,
        num_blocks=4
    )
    print(utils.get_num_parameters(net))
    outputs = net(inputs)
    print(outputs.shape)


if __name__ == '__main__':
    main()