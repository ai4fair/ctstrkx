#!/usr/bin/env python
# coding: utf-8

"""
Utilities for Processing the Overall Event:

The module contains useful functions for handling the data at the event level. 
More fine-grained utilities are reserved for `detector_utils` and `cell_utils`.
"""

# TODO: Pull module IDs out into a csv file for readability

from math import e
import os
import logging
import itertools
import torch
import trackml.dataset

import scipy as sp
import numpy as np
import pandas as pd

from torch_geometric.data import Data
from .heuristic_utils import get_layerwise_graph, get_all_edges, graph_intersection
from .graph_utils import get_input_edges
from .read_root_file import load_event

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_time_ordered_true_edges(hits):
    """Constructs the true edges using the MC Point time information for the hits.

    Args:
        hits (_type_): _description_

    Returns:
        _type_: The true edges in both directions
    """

    true_edges_start = []
    true_edges_end = []

    # Sort by the time of the MC Points
    for particle_id in hits.particle_id.unique():
        sortedHits = hits.query(f'particle_id=={particle_id}').sort_values('tT', ascending=True)
        for hit in range(sortedHits.x.size-1):
            true_edges_start.append(sortedHits.index[hit])
            true_edges_end.append(sortedHits.index[hit+1])

    true_edges = np.array([true_edges_start, true_edges_end])

    # Add the reverse edges
    true_edges = np.concatenate((true_edges, true_edges[[1,0]]), axis=1)

    return true_edges

    
def get_layerwise_edges(hits):
    """Get layerwise true edge list. Here 'hits' represent complete event."""
    
    # Sort by increasing distance from production (IP)
    hits = hits.assign(
        R=np.sqrt(
            (hits.x - hits.vx) ** 2 + (hits.y - hits.vy) ** 2 + (hits.z - hits.vz) ** 2
        )
    )
    hits = hits.sort_values("R").reset_index(drop=True).reset_index(drop=False)
    hits.loc[hits["particle_id"] == 0, "particle_id"] = np.nan
    hit_list = (
        hits.groupby(["particle_id", "layer_id"], sort=False)["index"]
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
    """Get modulewise (layerless) true edge list using the order
    of hits based on R. Here 'hits' represent complete event."""
    
    # Handle NaN and Null Values
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
    
    # Handle noise (particle_id == 0)
    signal.loc[signal["particle_id"] == 0, "particle_id"] = np.nan

    # Group by particle ID
    signal_list = signal.groupby(["particle_id"], sort=False)["index"].agg(
        lambda x: list(x)
    )
    
    # Build true edges from particle groups
    true_edges = []
    for row in signal_list.values:
        for i, j in zip(row[:-1], row[1:]):
            true_edges.append([i, j])

    true_edges = np.array(true_edges).T
    
    # Restore order
    true_edges = signal.unsorted_index.values[true_edges]
    return true_edges


def get_orderwise_edges(hits):
    """Get modulewise (layerless) true edge list using the order
    of occurrence of hits. Here 'hits' represent complete event."""
    
    # Handle NaN and Null Values    
    signal = hits[
        ((~hits.particle_id.isna()) & (hits.particle_id != 0)) & (~hits.vx.isna())
    ]
    signal = signal.drop_duplicates(
        subset=["particle_id", "volume_id", "layer_id", "module_id"]
    )
    
    # Reset index to get 'index' column
    signal = signal.reset_index(drop=False)
    
    # Preserve the Order
    # signal = signal.rename(columns={"index": "unsorted_index"}).reset_index(drop=False)

    # Handle Particle_id 0
    signal.loc[signal["particle_id"] == 0, "particle_id"] = np.nan
        
    # Group by particle ID and get list of indices of every particle (series of series).
    signal_list = signal.groupby(["particle_id"], sort=False)["index"].agg(
        lambda x: list(x)
    )
        
    # Generate Edges
    true_edges = []
    for row in signal_list.values:
        for i, j in zip(row[:-1], row[1:]):
            true_edges.append([i, j])
        
    # Return Edges
    true_edges = np.array(true_edges).T
    
    # Restore the Order
    # true_edges = signal.unsorted_index.values[true_edges]

    return true_edges


def process_particles(particles, selection=False):
    """
    Special manipulation on particles data frame
    """

    if selection:
        # just keep protons, pions, don't forget resetting index and dropping old one.
        particles = particles[particles['pdgcode'].isin([-2212, 2212, -211, 211])].reset_index(drop=True)
    
    return particles


def select_hits(event_file=None, file_reader=None, noise=False, skewed=False, readTruth=True, **kwargs):
    """
    Hit selection method from Exa.TrkX. Build a full event, select hits based on certain criteria.
    """
    
    if file_reader is not None:
    # load event (root or csv)
    if inputRootFile is not None:
        # load data using event_prefix (e.g. path/to/event0000000001)
        logging.info(f"Loading event {event_file} from ROOT file")
        hits, tubes, particles, truth = file_reader.load_event(int(event_file), read_truth=readTruth)
    else:
        # load data using ROOT file
        logging.info(f"Loading event {event_file} from CSV file")
        hits, tubes, particles, truth = trackml.dataset.load_event(event_file)
    
    # store original order of hits
    hits['original_order'] = hits.index

    # preprocess 'particles' to get nhits, and drop duplicates
    particles['nhits'] = particles.groupby(['particle_id'])['nhits'].transform('count')
    particles.drop_duplicates(inplace=True, ignore_index=True)

    # skip noise hits.
    if noise:
        # runs if noise=True
        truth = truth.merge(
            particles[["particle_id", "vx", "vy", "vz", "pdgcode"]], on="particle_id", how="left"
        )
    else:
        # runs if noise=False
        truth = truth.merge(
            particles[["particle_id", "vx", "vy", "vz", "pdgcode"]], on="particle_id", how="inner"
        )
    
    # derive new quantities from truth
    px = truth.tpx
    py = truth.tpy
    pz = truth.tpz

    # calculate pt, ptheta, peta, pphi
    pt = np.sqrt(px**2 + py**2)
    ptheta = np.arctan2(pt, pz)
    peta = -np.log(np.tan(0.5 * ptheta))
    pphi = np.arctan2(py, px)

    # assign pt, ptheta, peta, pphi to truth
    truth = truth.assign(pt=pt, ptheta=ptheta, peta=peta, pphi=pphi)
    
    # merge some columns of tubes to the hits
    hits = hits.merge(tubes[["hit_id", "isochrone", "skewed", "sector_id"]], on="hit_id")
    
    # skip skewed tubes
    if skewed is False:
        hits = hits.query('skewed==0')

    # Calculate derived variables from 'hits'
    r     = np.sqrt(hits.x**2 + hits.y**2)              # Transverse distance from the interaction point
    phi   = np.arctan2(hits.y, hits.x)                  # Azimuthal angle
    r3    = np.sqrt(hits.x**2 + hits.y**2 + hits.z**2)  # 3D distance from the interaction point
    theta = np.arccos(hits.z / r3)                      # Polar angle
    eta   = -np.log(np.tan(theta / 2.))                 # Pseudo-rapidity
    
    # Add r, phi, theta, eta to 'hits' and merge with 'truth'
    hits = hits.assign(r=r, phi=phi, theta=theta, eta=eta).merge(truth, on="hit_id")
    
    # FIXME: Check if Order is Changed
    # assert (hits['original_order'] == hits.index).all(), "Order disturbed after merging with truth"
    
    # Restore the original order
    hits = hits.sort_values(by='original_order').reset_index(drop=True)
    
    # Drop the original_order column as it is no longer needed
    hits = hits.drop(columns=['original_order'])
    
    # Add 'event_id' column to this event.
    hits = hits.assign(event_id=int(event_file[-10:]))

    return hits
    
    
# def build_event(feature_scale, inputedges, event_file=None, inputRootFile=None, layerwise=True, orderwise=False, modulewise=False, timeOrdered=True, noise=False, skewed=False, **kwargs):
def build_event(event_file, inputRootFile, feature_scale, 
                layerwise=True, modulewise=False, orderwise=False, timeOrdered=True, inputedges=True, 
                noise=False, skewed=False, **kwargs):
    """
    Builds the event data by loading the event file and preprocessing the hits data.

    Args:
        event_file (str): The path to the event file.
        feature_scale (float): The scale factor for the features.
        layerwise (bool, optional): Whether to build the layerwise true edges (default: True).
        modulewise (bool, optional): Whether to build the modulewise true edges (default: False).
        orderwise (bool, optional): Whether to build the orderwise true edges (default: False).
        inputedges (bool, optional): Whether to build the input edges (default: True).
        noise (bool, optional): Whether to include noise hits in the data (default: False).
        skewed (bool, optional): Whether to include skewed tubes in the data (default: False).
        **kwargs: Additional keyword arguments to be passed to the select_hits function.

    Returns:
        pandas.DataFrame: The preprocessed hits data containing the event_id column.

    Note:
        This function first calls the select_hits function to load the hits data and preprocess it.
        Then it calculates the derived variables from the hits data.
        Next, it merges the hits data with the truth data and adds the event_id column.
        Finally, it returns the preprocessed hits data.
    """
    
    # Load event using "event_file" prefix (load_event function transferred to select_hits function).
    # hits, tubes, particles, truth = trackml.dataset.load_event(event_file)
    
    # Select hits, add new/select columns, add event_id
    
    # TODO: Maybe can have dataformat flag here e.g. 
    # if dataformat = 'root' elif dataformat = 'csv' else raise error

    if file_reader is not None:
        hits = select_hits(event_file=event_file, file_reader=file_reader, noise=noise, skewed=skewed, readTruth=True, **kwargs).assign(event_id=int(event_file))
    else:
        hits = select_hits(event_file=event_file, noise=noise, skewed=skewed, **kwargs).assign(event_id=int(event_file[-10:]))
    
    # Get list of all layers
    layers = hits.layer.to_numpy()
    
    # Handle which truth graph(s) are being produced
    modulewise_true_edges, layerwise_true_edges = None, None
    orderwise_true_edges, time_ordered_true_edges = None, None
    
    # Get true edge list using the ordering of layers
    if layerwise:
        layerwise_true_edges, hits = get_layerwise_edges(hits)
        logging.info(
            "Layerwise truth graph built for {} with size {}".format(
                event_file, layerwise_true_edges.shape
            )
        )
    
    # Get true edge list without layer ordering
    if modulewise:
        modulewise_true_edges = get_modulewise_edges(hits)
        logging.info(
            "Modulewise truth graph built for {} with size {}".format(
                event_file, modulewise_true_edges.shape
            )
        )
    
    # Get true edge list without layer ordering (natural order)
    if orderwise:
        orderwise_true_edges = get_orderwise_edges(hits)
        logging.info(
            "Orederwise truth graph built for {} with size {}".format(
                event_file, orderwise_true_edges.shape

            )
        )

    # Get true edge list without layer ordering (time ordered)
    if timeOrdered:
        time_ordered_true_edges = get_time_ordered_true_edges(hits)
        logging.info(
            "Time ordered truth graph built for {} with size {}".format(
                event_file, time_ordered_true_edges.shape
                )
        )
    

    # Handle whether input graph(s) are being produced
    input_edges = None
    
    # Get input edge list using order of layers.
    if inputedges == "oldLayerwise":
        input_edges = get_layerwise_graph(hits, filtering=kwargs['filtering'], inneredges=False)  # without samelayer edges
        logging.info(
            "Layerwise input graph built for {} with size {}".format(
                event_file, input_edges.shape
                )
        )

    elif inputedges == "newLayerwise":
        input_edges = get_layerwise_graph(hits, filtering=kwargs['filtering'], inneredges=True)  # with samelayer edges
        logging.info(
            "Layerwise input graph built for {} with size {}".format(
                event_file, input_edges.shape
                )
        )

    elif inputedges == "all":
        input_edges = get_all_edges(hits)
        logging.info(
            "All input graph built for {} with size {}".format(
                event_file, input_edges.shape
                )
        )

    else:
        logging.error(f"{inputedges} is not a valid method to build input graphs")
        exit(1)

    # Get edge weight
    # TODO: No weights of tracks in STT data yet, skipping it.
    # Get edge weight
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
        orderwise_true_edges,
        time_ordered_true_edges,
        input_edges,
        hits["hit_id"].to_numpy(),
        hits.pt.to_numpy(),
        hits[["vx", "vy", "vz"]].to_numpy(),
        hits.pdgcode.to_numpy(),
        hits.ptheta.to_numpy(),
        hits.peta.to_numpy(),
        hits.pphi.to_numpy(),
    )
 
    
def prepare_event(event_file : str, inputedges : str, output_dir : str, file_reader = None ,modulewise=True, orderwise=True, timeOrdered=True, layerwise=True, noise=False, skewed=False, overwrite=False, **kwargs):
    """
    Main function for processing an event.

    Description:
        This function collects the hit, truth, edge and edge label information and saves them into a PyTorch geometry file.

    Args:
        event_file (str): Name of the event file
        inputedges (str): Specifies the method to build input edges. Options are "layerwise" or "all"
        output_dir (str): Directory in which to save the processed event
        modulewise (bool, optional): If true, true edges are constructed using the modulewise method. Defaults to True.
        timeOrdered (bool, optional): If true, true edges are constructed using the time ordered method. Defaults to True.
        layerwise (bool, optional): If true, true edges are constructed using the layerwise method. Defaults to True.
        noise (bool, optional): _description_. Defaults to False.
        skewed (bool, optional): _description_. Defaults to False.
        overwrite (bool, optional): _description_. Defaults to False.
    """

    try:
        if file_reader is not None:
            
            # feature scale for X=[r,phi,z]
            feature_scale = [100, np.pi, 100]

            # build event
            (
                X,
                pid,
                layers,
                layerwise_true_edges,
                modulewise_true_edges,
                time_ordered_true_edges,
                orderwise_true_edges,
                input_edges,
                hid,
                pt,
                vertex,
                pdgcode,
                ptheta,
                peta,
                pphi
            ) = build_event(
                feature_scale = feature_scale,
                file_reader = file_reader,
                event_file    = event_file,
                layerwise     = layerwise,
                modulewise    = modulewise,
                timeOrdered   = timeOrdered,
                orderwise     = orderwise,
                inputedges    = inputedges,
                noise         = noise,
                skewed        = skewed,
                **kwargs
            )

            filename = os.path.join(output_dir, event_file)
        
        else:
            evtid    = int(event_file[-10:])
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
                    time_ordered_true_edges,
                    orderwise_true_edges,
                    input_edges,
                    hid,
                    pt,
                    vertex,
                    pdgcode,
                    ptheta,
                    peta,
                    pphi
                ) = build_event(
                    event_file    = event_file,
                    feature_scale = feature_scale,
                    layerwise     = layerwise,
                    modulewise    = modulewise,
                    timeOrdered   = timeOrdered,
                    orderwise     = orderwise,
                    inputedges    = inputedges,
                    noise         = noise,
                    skewed        = skewed,
                    **kwargs
                )
            else:
                logging.info("{} already exists".format(evtid))
            
        # build pytorch_geometric Data module
        data = Data(
            event_file = event_file,
            x          = torch.from_numpy(X).float(),
            pid        = torch.from_numpy(pid),
            layers     = torch.from_numpy(layers),
            hid        = torch.from_numpy(hid),
            pt         = torch.from_numpy(pt),
            vertex     = torch.from_numpy(vertex),
            pdgcode    = torch.from_numpy(pdgcode),
            ptheta     = torch.from_numpy(ptheta),
            peta       = torch.from_numpy(peta),
            pphi       = torch.from_numpy(pphi)
        )
                
        # add edges to PyTorch Geometric Data module
        if layerwise_true_edges is not None:
            data.layerwise_true_edges = torch.from_numpy(layerwise_true_edges)

        if modulewise_true_edges is not None:
            data.modulewise_true_edges = torch.from_numpy(modulewise_true_edges)

        if time_ordered_true_edges is not None:
            data.time_ordered_true_edges = torch.from_numpy(time_ordered_true_edges)

        if orderwise_true_edges is not None:
            data.orderwise_true_edges = torch.from_numpy(orderwise_true_edges)
        
        # NOTE: I am jumping from Processing to GNN stage, so I need ground truth (GT) of input
        # edges (edge_index). After embedding, one gets GT as y, and after filtering one gets 
        # the GT in the form of 'y_pid'. As I intend to skip both the Embedding & the Filtering
        # stages, the input graph and its GT is build in Processing stage. The GNN can run after
        # either embedding or filtering stages so it look for either 'y' or 'y_pid', existence of
        # one of these means the execution of these stages i.e. if y_pid exists in data that means
        # both embedding and filtering stages has been executed. If only 'y' exists then only 
        # embedding stage has been executed. In principle, I should've only one of these in 'Data'.
        
        # Now, for my case, I will build input graph during Processing and also add its GT to the
        # data. If the 'edge_index' is build in Processing then ground truth (y or y_pid) should 
        # also be built here. The dimension of y (n) and y_pid (m) are given below, here m < n.
        
        # y (n): after embedding along with e_radius (2,n), y.shape==e_radius.shape[1]
        # y_pid (m): after filtering along with e_radius (2,m), y_pid.shape==e_radius.shape[1]
        
        # TODO: input_edges + true_edges [layerwise OR modulewise OR orderwise]
        if input_edges is not None:
            
            # select true edges
            if layerwise:
                true_edges = data.layerwise_true_edges
            elif modulewise:
                true_edges = data.modulewise_true_edges
            elif orderwise:
                true_edges = data.orderwise_true_edges
            elif timeOrdered:
                true_edges = data.time_ordered_true_edges
            else:
                true_edges = None
            
            assert true_edges is not None

            # get input graph with true edges
            input_edges = torch.from_numpy(input_edges)
            new_input_graph, y = graph_intersection(input_edges, true_edges)
            data.edge_index = new_input_graph
            # data.y = y     # if regime: [] will point to embedding
            data.y_pid = y  # if regime: [[pid]] points to filtering

        # add cell/tube information to Data, Check for STT
        # logging.info("Getting cell info")
        
        # if cell_information:
        #    data = get_cell_information(
        #        data, cell_features, detector_orig, detector_proc, endcaps, noise
        #    )

        with open(filename, "wb") as pickle_file:
            torch.save(data, pickle_file)

    except Exception as inst:
        logging.error("File:", event_file, "had exception", inst)
