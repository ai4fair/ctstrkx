#!/usr/bin/env python
# coding: utf-8

import torch
from torch_scatter import scatter_add
from torch.utils.checkpoint import checkpoint
from ..gnn_base import GNNBase
from ..utils.gnn_utils import make_mlp


class CheckpointedResAGNN(GNNBase):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        The model `ResAGNN` is the attention model with residual (aka "skip") connection.
        It was tested in "Performance of a geometric deep learning pipeline for HL-LHC
        particle tracking" [arXiv:2103.06995] by Exa.TrkX. No other study exist so far.
        """

        hparams["output_activation"] = (
            None if "output_activation" not in hparams else hparams["output_activation"]
        )
        hparams["batchnorm"] = (
            False if "batchnorm" not in hparams else hparams["batchnorm"]
        )

        # Setup input network
        self.input_network = make_mlp(
            (hparams["in_channels"]),
            [hparams["hidden"]] * hparams["nb_node_layer"],
            hidden_activation=hparams["hidden_activation"],
            output_activation=hparams["output_activation"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
        )

        # Setup edge network
        self.edge_network = make_mlp(
            (hparams["in_channels"] + hparams["hidden"]) * 2,
            [hparams["in_channels"] + hparams["hidden"]] * hparams["nb_edge_layer"]
            + [1],
            hidden_activation=hparams["hidden_activation"],
            output_activation=hparams["output_activation"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
        )

        # Setup node network
        self.node_network = make_mlp(
            (hparams["in_channels"] + hparams["hidden"]) * 2,
            [hparams["hidden"]] * hparams["nb_node_layer"],
            hidden_activation=hparams["hidden_activation"],
            output_activation=hparams["output_activation"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
        )

    def forward(self, x, edge_index):
        start, end = edge_index
        input_x = x

        x = self.input_network(x)

        # Shortcut connect the inputs onto the hidden representation
        x = torch.cat([x, input_x], dim=-1)

        # Loop over iterations of edge and node networks
        for i in range(self.hparams["n_graph_iters"]):
            x_inital = x

            # Apply edge network
            edge_inputs = torch.cat([x[start], x[end]], dim=1)
            e = checkpoint(self.edge_network, edge_inputs)
            e = torch.sigmoid(e)

            # Apply node network
            messages = scatter_add(
                e * x[start], end, dim=0, dim_size=x.shape[0]
            ) + scatter_add(
                e * x[end], start, dim=0, dim_size=x.shape[0]
            )

            node_inputs = torch.cat([messages, x], dim=1)
            x = checkpoint(self.node_network, node_inputs)

            # Shortcut connect the inputs onto the hidden representation
            x = torch.cat([x, input_x], dim=-1)

            # Residual connection
            x = x_inital + x

        edge_inputs = torch.cat([x[start], x[end]], dim=1)
        return checkpoint(self.edge_network, edge_inputs)