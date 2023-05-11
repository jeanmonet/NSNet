import torch
import torch.nn as nn
import math

from models.mlp import MLP

# pytorch-scatter ported to pytorch main
# https://github.com/rusty1s/pytorch_scatter/issues/241#issuecomment-1336116049
# from torch_scatter import scatter_sum, scatter_logsumexp

from utils.scatter import scatter_sum
from utils.scatter import scatter_logsumexp
# from torch.scatter_reduce import scatter_sum
# from torch.scatter_reduce import scatter_logsumexp


class NSNet(nn.Module):
    def __init__(self, opts):
        super(NSNet, self).__init__()
        self.opts = opts
        self.c2l_edges_init = nn.Parameter(torch.randn(1, self.opts.dim))   # clause to literal edges
        self.l2c_edges_init = nn.Parameter(torch.randn(1, self.opts.dim))   # literal to clause edges
        self.denom = math.sqrt(self.opts.dim)
        # Clause node to assignment node message update: m_{a, i}(x_i) = MLP(i, a)
        self.c2l_msg_update = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation)
        # Assignment node to clause node message update: m_{i, a}(x_i) = MLP(a, i)
        self.l2c_msg_update = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation)
        self.l2c_msg_norm = MLP(self.opts.n_mlp_layers, self.opts.dim * 2, self.opts.dim, self.opts.dim, self.opts.activation)
        self.c_readout = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, 1, self.opts.activation)
        self.l_readout = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, 1, self.opts.activation)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, data):
        """
        Explanation of what this function does:
        1. Initialize the clause to literal edges and literal to clause edges
        2. For each round:
            2.1. Compute the clause to literal message
            2.2. Compute the literal to clause message
            2.3. Compute the clause to literal message aggregation
            2.4. Compute the clause to literal message normalization
        3. Compute the clause readout
        4. Compute the literal readout
        5. Compute the clause to literal message aggregation
        6. Compute the clause to literal message normalization
        7. Compute the clause to literal message
        8. Compute the literal to clause message
        9. Compute the clause to literal message aggregation
        10. Compute the clause to literal message normalization
        ...

        Explain what the readout is:
        The readout is the final output of the model. It is a scalar value that is computed from the clause and literal
        embeddings. The clause readout is computed by summing the clause embeddings and passing them through a MLP. The
        literal readout is computed by summing the literal embeddings and passing them through a MLP.

        """
        l_size = data.l_size.sum().item()        # number of literals
        c_size = data.c_size.sum().item()        # number of clauses
        num_edges = data.num_edges

        sign_l_edge_index = data.sign_l_edge_index
        c2l_msg_repeat_index = data.c2l_msg_repeat_index
        c2l_msg_scatter_index = data.c2l_msg_scatter_index

        l2c_msg_aggr_repeat_index = data.l2c_msg_aggr_repeat_index
        l2c_msg_aggr_scatter_index = data.l2c_msg_aggr_scatter_index
        l2c_msg_scatter_index = data.l2c_msg_scatter_index

        if self.opts.task == 'model-counting':
            c_blf_repeat_index = data.c_blf_repeat_index
            c_blf_scatter_index = data.c_blf_scatter_index
            c_blf_norm_index = data.c_blf_norm_index
            v_degrees = data.v_degrees
            c_batch = data.c_batch
            v_batch = data.v_batch
            c_bethes = []
            v_bethes = []

        c2l_edges_feat = (self.c2l_edges_init / self.denom).repeat(num_edges, 1)
        l2c_edges_feat = (self.l2c_edges_init / self.denom).repeat(num_edges, 1)

        for _ in range(self.opts.n_rounds):
            c2l_msg = scatter_sum(c2l_edges_feat[c2l_msg_repeat_index], c2l_msg_scatter_index, dim=0, dim_size=num_edges)
            l2c_edges_feat_new = self.l2c_msg_update(c2l_msg)
            v2c_edges_feat_new = l2c_edges_feat_new.reshape(num_edges // 2, -1)
            pv2c_edges_feat_new, nv2c_edges_feat_new = torch.chunk(v2c_edges_feat_new, 2, 1)
            l2c_edges_feat_inv = torch.cat([nv2c_edges_feat_new, pv2c_edges_feat_new], dim=1).reshape(num_edges, -1)
            l2c_edges_feat = self.l2c_msg_norm(torch.cat([l2c_edges_feat_new, l2c_edges_feat_inv], dim=1))

            l2c_msg_aggr = scatter_sum(l2c_edges_feat[l2c_msg_aggr_repeat_index], l2c_msg_aggr_scatter_index, dim=0, dim_size=l2c_msg_scatter_index.shape[0])
            l2c_msg = scatter_logsumexp(l2c_msg_aggr, l2c_msg_scatter_index, dim=0, dim_size=num_edges)
            c2l_edges_feat = self.c2l_msg_update(l2c_msg)

        if self.opts.task == 'model-counting':
            c_blf_aggr = scatter_sum(l2c_edges_feat[c_blf_repeat_index], c_blf_scatter_index, dim=0, dim_size=c_blf_norm_index.shape[0])
            c_blf_aggr = self.c_readout(c_blf_aggr)
            c_blf_norm = scatter_logsumexp(c_blf_aggr, c_blf_norm_index, dim=0, dim_size=c_size)
            c_norm_blf = c_blf_aggr - c_blf_norm[c_blf_norm_index]
            c_bethe = -scatter_sum(c_norm_blf * c_norm_blf.exp(), c_blf_norm_index, dim=0, dim_size=c_size).reshape(-1)

            l_blf_aggr = scatter_sum(c2l_edges_feat, sign_l_edge_index, dim=0, dim_size=l_size)
            l_blf_aggr = self.l_readout(l_blf_aggr)
            v_blf_aggr = l_blf_aggr.reshape(-1, 2)
            v_blf_norm = torch.logsumexp(v_blf_aggr, dim=1, keepdim=True)
            v_norm_blf = v_blf_aggr - v_blf_norm
            v_bethe = (v_degrees - 1) * ((v_norm_blf * v_norm_blf.exp()).sum(dim=1))

            return scatter_sum(c_bethe, c_batch, dim=0, dim_size=data.l_size.shape[0]) + \
                scatter_sum(v_bethe, v_batch, dim=0, dim_size=data.l_size.shape[0])
        else:
            l_logit = scatter_sum(c2l_edges_feat, sign_l_edge_index, dim=0, dim_size=l_size)
            l_logit = self.l_readout(l_logit)
            v_logit = l_logit.reshape(-1, 2)
            # TEMP - check NaNs in model output
            # res = self.softmax(v_logit)
            # if res.isnan().sum() > 0:
            #     raise ValueError("WE HAVE NANS IN FWD")
            # return res
            return self.softmax(v_logit)
