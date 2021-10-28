import jax
import jax.numpy as jnp
import haiku as hk
import optax
import numpy as np

import functools
import logging
from typing import Optional, Mapping, Any
from functools import partial 

class DenseBlock(hk.Module):
  def __init__(
      self, 
      nb_layers:int,
      hidden_dim:int,
      name: Optional[str]= None):
    super().__init__(name=name)

    layers = []

    assert nb_layers > 0, 'Number of layer must greater than zero'

    for i in range(nb_layers - 1):
      layers += [hk.Linear(hidden_dim,), jax.nn.relu]
    layers.append(hk.Linear(1))

    self.mlp = hk.Sequential(layers)
  
  def __call__(
      self,
      data: Mapping[str, jnp.ndarray],
  )-> jnp.ndarray:
    out = self.mlp(data['images'])
    return out