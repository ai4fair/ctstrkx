#!/usr/bin/env python
# coding: utf-8

# From Exa.TrkX NeurIPS19
import itertools
def pairwise(iterable):
  """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
  a, b = itertools.tee(iterable)
  next(b, None)
  return zip(a, b)
  

# Connected Components Labeling
from .ccl.cc import tracks_from_cc

# Path Finding
from .wrangler.wrangler import get_tracks





# SttCSVReader Class (Alt. to Event Class)
from .reader import SttCSVReader, Draw_Reader_Event

# Event Class 
from .event import Event, Compose_Event, Draw_Compose_Event










