#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
from torch_scatter import scatter_add
from ..gnn_base import GNNBase
from ..utils.gnn_utils import make_mlp


class EdgeNetwork(nn.Module):
    """
    A module which computes weights for edges of the graph.
    For each edge, it selects the associated nodes' features
    and applies some fully-connected network layers with a final
    sigmoid activation.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        nb_layers,
        hidden_activation="Tanh",
        layer_norm=True,
    ):
        super(EdgeNetwork, self).__init__()
        self.network = make_mlp(
            input_dim * 2,
            [hidden_dim] * nb_layers + [1],
            hidden_activation=hidden_activation,
            output_activation=None,
            layer_norm=layer_norm,
        )

    def forward(self, x, edge_index):
        # Select the features of the associated nodes
        start, end = edge_index
        edge_inputs = torch.cat([x[start], x[end]], dim=1)
        return self.network(edge_inputs).squeeze(-1)


class NodeNetwork(nn.Module):
    """
    A module which computes new node features on the graph.
    For each node, it aggregates the neighbor node features
    (separately on the input and output side), and combines
    them with the node's previous features in a fully-connected
    network to compute the new features.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        nb_layers,
        hidden_activation="Tanh",
        layer_norm=True,
    ):
        super(NodeNetwork, self).__init__()
        self.network = make_mlp(
            input_dim * 2,
            [output_dim] * nb_layers,
            hidden_activation=hidden_activation,
            output_activation=None,
            layer_norm=layer_norm,
        )

    def forward(self, x, e, edge_index):
        start, end = edge_index
        
        # Original Aggregation Opertion (i.e. CTD 2018):
        # Aggregate edge-weighted edges going in both directions and handle them separately
        # mi = scatter_add(e[:, None] * x[start], end, dim=0, dim_size=x.shape[0])
        # mo = scatter_add(e[:, None] * x[end], start, dim=0, dim_size=x.shape[0])
        # node_inputs = torch.cat([mi, mo, x], dim=1)
        
        # New Aggregation Operation:
        # Aggregate edge-weighted going in both directions and sum them
        messages = scatter_add(
            e[:, None] * x[start], end, dim=0, dim_size=x.shape[0]
        ) + scatter_add(e[:, None] * x[end], start, dim=0, dim_size=x.shape[0])
        node_inputs = torch.cat([messages, x], dim=1)
        return self.network(node_inputs)


class ResAGNN(GNNBase):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        The model `ResAGNN` is the attention model with residual (aka "skip") connection.
        It was tested in "Performance of a geometric deep learning pipeline for HL-LHC
        particle tracking" [arXiv:2103.06995] by Exa.TrkX. No other study exist so far.
        """

        # Setup input network
        self.input_network = make_mlp(
            hparams["spatial_channels"],
            [hparams["hidden"]] * hparams["nb_node_layer"],
            output_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
        )

        # Setup edge network
        self.edge_network = EdgeNetwork(
            hparams["spatial_channels"] + hparams["hidden"],   # formerly in_channels
            hparams["spatial_channels"] + hparams["hidden"],
            hparams["nb_edge_layer"],
            hparams["hidden_activation"],
            hparams["layernorm"],
        )
        
        # Setup node layers
        self.node_network = NodeNetwork(
            hparams["spatial_channels"] + hparams["hidden"],
            hparams["hidden"],
            hparams["nb_node_layer"],
            hparams["hidden_activation"],
            hparams["layernorm"],
        )

    def forward(self, x, edge_index):
        input_x = x

        x = self.input_network(x)

        # Shortcut connect the inputs onto the hidden representation
        x = torch.cat([x, input_x], dim=-1)

        # Loop over iterations of edge and node networks
        for i in range(self.hparams["n_graph_iters"]):
            x_inital = x

            # Apply edge network
            e = torch.sigmoid(self.edge_network(x, edge_index))

            # Apply node network
            x = self.node_network(x, e, edge_index)

            # Shortcut connect the inputs onto the hidden representation
            x = torch.cat([x, input_x], dim=-1)

            # Residual connection
            x = x_inital + x

        return self.edge_network(x, edge_index)
