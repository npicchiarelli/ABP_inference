'''
Modified from https://gitlab.ethz.ch/cmbm-public/toolboxes/pignpi.git Copyright (c) 2022 CMBM-public

'''

import numpy as np
import torch
from torch import nn
from torch.functional import F
from torch.optim import Adam
from torch_geometric.nn import MetaLayer, MessagePassing # type: ignore
from torch.nn import Sequential as Seq, Linear as Lin, SiLU
from torch.autograd import Variable, grad

torch.set_default_dtype(torch.float64)

# PIGNPI with SiLU (default)
class GN_force_SiLU(MessagePassing):
    def __init__(self, n_node, n_f, msg_dim, ndim, hidden=300, aggr='add', flow="target_to_source"):
        super(GN_force_SiLU, self).__init__(aggr=aggr, flow=flow)  # "Add" aggregation.

        torch.set_default_dtype(torch.float64)
        
        self.ndim = ndim
        self.n_node = n_node
        print("In GN_force_SiLU, n_node is : {}".format(self.n_node) )

        self.msg_fnc_type0 = Seq(
            Lin(2*n_f, hidden), # pos, vel, charge, mass; for two nodes
            SiLU(),

            Lin(hidden, hidden),
            SiLU(),

            Lin(hidden, hidden),
            SiLU(),

            #(Can turn on or off this layer:)
            Lin(hidden, hidden), 
            SiLU(),
            
            Lin(hidden, msg_dim)
        )

        self.node_fnc = Seq(
            Lin(msg_dim+n_f, hidden),
            SiLU(),
            Lin(hidden, hidden),
            SiLU(),
            Lin(hidden, hidden),
            SiLU(),
#             Lin(hidden, hidden),
#             SiLU(),
            Lin(hidden, ndim)
        )

    def forward(self, x, edge_index, edge_feature):
        #x is [n, n_f]

        if edge_feature == None:
            predicted_force = self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, edge_feature_list = None)
        else:
            predicted_force = self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, edge_feature_list = edge_feature)
        
        predicted_acc = predicted_force / x[:, -1][:, None] # add edges from all types, then divide the mass
        return predicted_acc
      

    def message(self, x_i, x_j, edge_feature_list):
        if edge_feature_list == None:
            # x_i has shape [n_e, n_f]; x_j has shape [n_e, n_f]
            tmp = torch.cat([x_i, x_j], dim=1)  # tmp has shape [E, 2 * in_channels]
        else:
            # x_i has shape [n_e, n_f]; x_j has shape [n_e, n_f]
            # edge_feature_list has shape [n_e, n_ef]
            tmp = torch.cat([x_i, x_j, edge_feature_list], dim=1)
        
        return self.msg_fnc_type0(tmp)
        
    
    def update(self, aggr_out, x=None):
        # aggr_out has shape [n, msg_dim]
        # tmp = torch.cat([x, aggr_out], dim=1)
        # return self.node_fnc(tmp) / x[:, -1][:, None]
        
        return aggr_out # not dividing mass here
