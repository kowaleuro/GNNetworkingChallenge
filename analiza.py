import tensorflow as tf
import os
import numpy as np
import networkx as nx
import pprint

ds_path = 'data/data_mb_cv'
dataset = tf.data.Dataset.load(f"{ds_path}/{0}/training", compression="GZIP")

for element in dataset:  
  print(element)
  break



