#!/usr/bin/env python
# coding: utf-8

# SttCSVReader Class (Alt. to Event Class)
from .reader import SttCSVReader, Draw_Reader_Event

# Event Class 
from .event import Compose_Event, Draw_Compose_Event


# From Exa.TrkX NeurIPS19
import itertools
def pairwise(iterable):
  """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
  a, b = itertools.tee(iterable)
  next(b, None)
  return zip(a, b)






