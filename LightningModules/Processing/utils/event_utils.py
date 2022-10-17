#!/usr/bin/env python
# coding: utf-8

"""
Utilities for Processing the Overall Event:

The module contains useful functions for handling the data at the event level. 
More fine-grained utilities are reserved for `detector_utils` and `cell_utils`.
"""

import os
import logging
import itertools

import scipy as sp
import numpy as np
import pandas as pd
import trackml.dataset

import torch
from torch_geometric.data import Data

from .graph_utils import get_input_edges, graph_intersection

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_layerwise_edges(hits):
    """Build Layerwise True Edges i.e. the True Graph. Here `hits` represent complete event."""
    
    # ADAK: Sort by increasing distance from production (IP)
    hits = hits.assign(
        R=np.sqrt(
            (hits.x - hits.vx) ** 2 + (hits.y - hits.vy) ** 2 + (hits.z - hits.vz) ** 2
        )
    )
    hits = hits.sort_values("R").reset_index(drop=True).reset_index(drop=False)
    hits.loc[hits["particle_id"] == 0, "particle_id"] = np.nan
    hit_list = (
        hits.groupby(["particle_id", "layer_id"], sort=False)["index"]  # ADAK: layer >> layer_id
        .agg(lambda x: list(x))
        .groupby(level=0)
        .agg(lambda x: list(x))
    )

    true_edges = []
    for row in hit_list.values:
        for i, j in zip(row[0:-1], row[1:]):
            true_edges.extend(list(itertools.product(i, j)))
            
    true_edges = np.array(true_edges).T
    return true_edges, hits
 
    
def get_modulewise_edges(hits):
    """Get modulewise (layerless) true edge list. Here hits represent complete event."""
    signal = hits[
        ((~hits.particle_id.isna()) & (hits.particle_id != 0)) & (~hits.vx.isna())
    ]
    signal = signal.drop_duplicates(
        subset=["particle_id", "volume_id", "layer_id", "module_id"]
    )

    # Sort by increasing distance from production
    signal = signal.assign(
        R=np.sqrt(
            (signal.x - signal.vx) ** 2
            + (signal.y - signal.vy) ** 2
            + (signal.z - signal.vz) ** 2
        )
    )
    signal = signal.sort_values("R").reset_index(drop=False)

    # Handle re-indexing
    signal = signal.rename(columns={"index": "unsorted_index"}).reset_index(drop=False)
    signal.loc[signal["particle_id"] == 0, "particle_id"] = np.nan

    # Group by particle ID
    signal_list = signal.groupby(["particle_id"], sort=False)["index"].agg(
        lambda x: list(x)
    )
    
    # Form edges
    true_edges = []
    for row in signal_list.values:
        for i, j in zip(row[:-1], row[1:]):
            true_edges.append([i, j])

    true_edges = np.array(true_edges).T

    true_edges = signal.unsorted_index.values[true_edges]

    return true_edges


def select_hits(event_prefix=None, skewed=False, noise=False, min_pt=None):
    """Hit selection, merge info into a single dataframe"""
    
    hits, tubes, particles, truth = trackml.dataset.load_event(event_prefix)
    
    if noise:
        truth = truth.merge(
            particles[["particle_id", "vx", "vy", "vz"]], on="particle_id", how="left"
        )
    else:
        truth = truth.merge(
            particles[["particle_id", "vx", "vy", "vz"]], on="particle_id", how="inner"
        )
    
    truth = truth.assign(pt=np.sqrt(truth.tpx**2 + truth.tpy**2))
    
    if min_pt:
        truth = truth[truth.pt > min_pt]
    
    
    # ADAK: Start
    
    # Need 'isochrone', 'skewed' & 'sector_id' columns from 'tubes'
    hits = hits.merge(tubes[["hit_id", "isochrone", "skewed", "sector_id"]], on="hit_id")
    
    # Handle Skewed Layers
    if skewed is False:
       
        # filter
        hits = hits.query('skewed==0')
        
        # reassign layer_ids from 0,1,2...,17
        vlids = hits.layer_id.unique()
        n_det_layers = hits.layer_id.unique().shape[0]
        vlid_groups = hits.groupby(['layer_id'])
        hits = pd.concat([vlid_groups.get_group(vlids[i]).assign(layer=i) for i in range(n_det_layers)])
    else:
        hits = hits.rename(columns={"layer_id": "layer"})
    
    # ADAK: End
    
    # Calculate derived hits variables
    r = np.sqrt(hits.x**2 + hits.y**2)
    phi = np.arctan2(hits.y, hits.x)
    
    # Select the data columns we need
    hits = hits.assign(r=r, phi=phi).merge(truth, on="hit_id")
    
    return hits


def build_event(event_prefix, feature_scale, layerwise=True, modulewise=True,
                inputedges=False, skewed=False, noise=False, min_pt=None):
    """True edges using ordering by R'=distance from production vertex of each particle.
    Return: [X=(r, phi, z), pid, layers, layerwise, modulewise, inputedges, hid, pt]"""
    
    # Load event using "event_prefix"
    # hits, tubes, particles, truth = trackml.dataset.load_event(event_prefix)
    
    # Select hits, add new/select columns, add event_id
    hits = select_hits(event_prefix, skewed, noise, min_pt).assign(
        event_id=int(event_prefix[-10:])
    )
    
    # Get list of all layers
    layers = hits.layer.to_numpy()

    # Handle which truth graph(s) are being produced
    modulewise_true_edges, layerwise_true_edges = None, None
    
    # Get true edge list using the ordering of layers
    if layerwise:
        layerwise_true_edges, hits = get_layerwise_edges(hits)
        logging.info(
            "Layerwise truth graph built for {} with size {}".format(
                event_prefix, layerwise_true_edges.shape
            )
        )
    
    # Get true edge list without layer ordering
    if modulewise:
        modulewise_true_edges = get_modulewise_edges(hits)
        logging.info(
            "Modulewise truth graph built for {} with size {}".format(
                event_prefix, modulewise_true_edges.shape
            )
        )
    
    # Handle whether input graph(s) are being produced
    layerwise_input_edges = None
    
    # Get input edge list using order of layers.
    if inputedges:
        layerwise_input_edges = get_input_edges(hits)
        logging.info(
            "Layerwise input graph built for {} with size {}".format(
                event_prefix, layerwise_input_edges.shape
            )
        )

    # Get edge weight
    # TODO: No weights of tracks in STT data yet, skipping it.
    
    # edge_weights = (
    #    hits.weight.to_numpy()[modulewise_true_edges]
    #    if modulewise
    #    else hits.weight.to_numpy()[layerwise_true_edges]
    # )
    # edge_weight_average = (edge_weights[0] + edge_weights[1]) / 2
    # edge_weight_norm = edge_weight_average / edge_weight_average.mean()

    logging.info("Weights are not constructed, no weights for STT")

    return (
        hits[["r", "phi", "isochrone"]].to_numpy() / feature_scale,
        hits.particle_id.to_numpy(),
        layers,
        layerwise_true_edges,
        modulewise_true_edges,
        layerwise_input_edges,
        hits["hit_id"].to_numpy(),
        hits.pt.to_numpy(),
        # edge_weight_norm,
    )
 
    
def prepare_event(
    event_file,
    output_dir=None,
    modulewise=True,
    layerwise=True,
    inputedges=True,
    skewed=False,
    noise=False,
    min_pt=None,
    overwrite=False,
    **kwargs
):

    """Prepare an event when called in FeatureStore Module"""
    try:
        evtid = int(event_file[-10:])
        filename = os.path.join(output_dir, str(evtid))

        if not os.path.exists(filename) or overwrite:
            logging.info("Preparing event {}".format(evtid))
            
            # feature scale for X=[r,phi,z]
            feature_scale = [100, np.pi, 100]
            
            # build event
            (
                X,
                pid,
                layers,
                layerwise_true_edges,
                modulewise_true_edges,
                layerwise_input_edges,
                hid,
                pt,
                # weights,
            ) = build_event(
                event_prefix=event_file,
                feature_scale=feature_scale,
                layerwise=layerwise,
                modulewise=modulewise,
                inputedges=inputedges,
                skewed=skewed,
                noise=noise,
                min_pt=min_pt
            )
            
            # build pytorch_geometric Data module
            data = Data(
                x=torch.from_numpy(X).float(),
                pid=torch.from_numpy(pid),
                layers=torch.from_numpy(layers),
                event_file=event_file,
                hid=torch.from_numpy(hid),
                pt=torch.from_numpy(pt),
                # weights=torch.from_numpy(weights),
            )
            
            # add edges to pytorch_geometric Data module
            if modulewise_true_edges is not None:
                data.modulewise_true_edges = torch.from_numpy(modulewise_true_edges)
                
            if layerwise_true_edges is not None:
                data.layerwise_true_edges = torch.from_numpy(layerwise_true_edges)
                       
            if layerwise_input_edges is not None:
                input_edges = torch.from_numpy(layerwise_input_edges)
                new_input_graph, y = graph_intersection(input_edges, data.layerwise_true_edges)
                data.edge_index = new_input_graph
                data.y = y

            # TODO: add cell/tube information to Data, Check for STT
            # logging.info("Getting cell info")
            
            # if cell_information:
            #    data = get_cell_information(
            #        data, cell_features, detector_orig, detector_proc, endcaps, noise
            #    )

            with open(filename, "wb") as pickle_file:
                torch.save(data, pickle_file)

        else:
            logging.info("{} already exists".format(evtid))
    except Exception as inst:
        print("File:", event_file, "had exception", inst)
