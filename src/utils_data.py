#!/usr/bin/env python
# coding: utf-8
import os
import glob
import pandas as pd
# import dask.dataframe as dd


# Preprocessing Tasks
def preprocess(hits):
    """Function to perform certain tasks once an event is loaded"""
    # Start: Preprocessing Tasks
    # hits = hits.drop(['z', 'tube_id','depcharge', 'tx', 'ty', 'tz','px','py','pz'], axis=1)
    # hits = event[['skewed', 'layer_id']]       # select only 2 columns
    # hits = hits[hits['skewed'] == 0]           # all non skewed
    # hits = hits.query('skewed==0')
    # hits = hits[hits['skewed'] == 1]           # all skewed, layers = 8,9,2,...15 are skewed
    # hits = hits.query('skewed==1')
    # hits = hits[hits['layer_id'] < 8]          # layers = 0,1,2,...7 are before skewed
    # hits = hits[hits['layer_id'] > 15]         # layers = 16,17,...24 are after skewed
    # hits = hits[(hits['layer_id'] > 7) &
    # (hits['layer_id'] < 16)]                   # layers = 8,9,2,...15 are skewed
    # hits.cluster_id.unique()
    # hits.layer.unique()
    # hits[hits["layer"] < 3].head()
    # hits[(hits["layer"] < 5) & (hits["layer"] > 3)].head()
    # hits.loc[hits["layer"].isin([2, 3])].head()
    # hits.loc[~hits["layer"].isin([2, 3])].head()
    # hits.loc[(hits["cluster_id"]==8) & hits["layer"].isin([2, 3])].head()
    return hits


# Load single event using event_prefix
def get_event(path=None, event_id=0):
    """Load Single CSV/Event File"""
    
    event_prefix = str('event{!s}'.format(format(event_id, '010d')))
    prefix = path + event_prefix
    
    hits = pd.read_csv(prefix + '-hits.csv')
    tubes = pd.read_csv(prefix + '-tubes.csv')
    particles = pd.read_csv(prefix + '-particles.csv')
    truth = pd.read_csv(prefix + '-truth.csv')
    
    return hits, tubes, particles, truth


# Load all events using Pandas
def pandas_events(path="None"):
    """Load multiple CSV files into Pandas"""
    # map(function, iterable) == map(pd.read_csv, files) where variable
    # the 'files = glob.glob(os.path.join(path+"*-hits.csv"))' is a list
    hits = pd.concat(map(pd.read_csv, glob.glob(os.path.join(path + "*-hits.csv"))))
    tubes = pd.concat(map(pd.read_csv, glob.glob(os.path.join(path + "*-tubes.csv"))))
    particles = pd.concat(map(pd.read_csv, glob.glob(os.path.join(path + "*-particles.csv"))))
    truth = pd.concat(map(pd.read_csv, glob.glob(os.path.join(path + "*-truth.csv"))))

    return hits, tubes, particles, truth


# Load all events using Dask
def dask_events(path="None"):
    """Load multiple CSV files into Dask"""
    pass
    
    # load multiple files at once
    # hits = dd.read_csv(path + '*-hits.csv')
    # truth = dd.read_csv(path + '*-truth.csv')
    # particles = dd.read_csv(path + '*-particles.csv')
    # tubes = dd.read_csv(path + '*-tubes.csv')

    # convert to pandas format, if required.
    # hits = hits.compute()
    # truth = truth.compute()
    # particles = particles.compute()
    # tubes = tubes.compute()

    # return hits, tubes, particles, truth


def get_geometry(events=None):
    """
    Pass all events, it will extract (x,y,z) values based on
    unique tube_ids. Sort it, dump it as a csv file.
    """
    df = events.drop_duplicates(subset='tube_id', keep="first")
    df = df[['x', 'y', 'z', 'skewed', 'tube_id', 'layer_id', 'sector_id']]
    df = df.sort_values(['tube_id', 'layer_id'])
    df.to_csv('stt.csv', encoding='utf-8', index=False)
