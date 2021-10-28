import jax
import jax.numpy as jnp
import haiku as hk
import optax
import numpy as np

import functools
import logging
from typing import Optional, Mapping, Any
from functools import partial 

from _src.models import DenseBlock


def build_forward_fn(hidden_dim: int, nb_layers: int):
    """Create the model's forward pass."""

    def forward_fn(data: Mapping[str, jnp.ndarray],
                   is_training: bool = True) -> jnp.ndarray:
        mlp  = DenseBlock(nb_layers, hidden_dim)
        logits = mlp(data)
        return logits
    return forward_fn
    
def bce_loss(w,x,y):
    max_val = jnp.clip(x, 0, None)
    x = w * x
    loss = x - x * y + max_val + jnp.log(jnp.exp(-max_val) + jnp.exp((-x - max_val)))
    loss = jnp.mean(loss)
    return loss


def loss_fn(
    forward_fn,
    params,
    rng,
    data: Mapping[str, jnp.ndarray],
    w:int,
    is_trainning:bool = True
) -> jnp.ndarray:

  logits = forward_fn(params,rng, data, is_trainning)
  labels = data['labels']
  l2_loss = sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params))
  x = jax.nn.sigmoid(logits)
  y = labels
  loss = bce_loss(1,x,y)
  penalty = jax.grad(bce_loss)(1.0,x,y)
  penalty = jnp.sum(penalty**2)

  loss = (loss +0.000810794568*l2_loss + w * penalty)/w
  
  return loss





class GradientUpdater:
    """A stateless abstraction around an init_fn/update_fn pair.
    This extracts some common boilerplate from the training loop.
    """

    def __init__(self, net_init, loss_fn,
                 optimizer: optax.GradientTransformation):
        self._net_init = net_init
        self._loss_fn = loss_fn
        self._opt = optimizer

    @functools.partial(jax.jit, static_argnums=0)
    def init(self, master_rng, data):
        """Initializes state of the updater."""
        out_rng, init_rng = jax.random.split(master_rng)
        params = self._net_init(init_rng, data)
        opt_state = self._opt.init(params)
        out = dict(
            step=np.array(0),
            rng=out_rng,
            opt_state=opt_state,
            params=params,
        )
        return out
    
    def p_loss(self, params, rng, data1, data2, w):
      loss1 = self._loss_fn(params, rng, data1, w)
      loss2 = self._loss_fn(params, rng, data2, w)
      return (loss1 + loss2)/2
   
    @functools.partial(jax.jit, static_argnums=0)
    def update(self, state: Mapping[str, Any], data1, data2, w):
        """Updates the state using some data and returns metrics."""
        rng, new_rng = jax.random.split(state['rng'])
        params = state['params']
        loss, g = jax.value_and_grad(self.p_loss)(params, rng, data1, data2, w)
        

        updates, opt_state = self._opt.update(g, state['opt_state'])
        params = optax.apply_updates(params, updates)
       

        new_state = {
            'step': state['step'] + 1,
            'rng': new_rng,
            'opt_state': opt_state,
            'params': params,
        }

        metrics = {
            'step': state['step'],
            'loss': loss,
        }
        return new_state, metrics


def accuracy(predictions:jnp.ndarray, labels:jnp.ndarray) -> jnp.ndarray:
  predictions = (predictions > 0.0)
  result = jnp.mean(predictions== labels )
  return result

def evaluate(params, rng, forward_fn, data):
  
  logits = forward_fn.apply(params, rng, data)
  avg_accuracy = accuracy(logits, data['labels'])
  
  return avg_accuracy





       
       

  