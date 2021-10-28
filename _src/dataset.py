import tensorflow_datasets as tfds
import tensorflow as tf

import jax
import jax.numpy as jnp
import haiku as hk
import optax
import numpy as np

import functools
import logging
from typing import Optional, Mapping, Any
from functools import partial 

class Environments:
  def __init__(self, train_data:[Mapping[str,np.ndarray]], test_data:[Mapping[str,np.ndarray]]):
    self.train_data = train_data
    self.test_data = test_data




def get_mnist():
  data_dir = '/tmp/tfds'
  # Fetch full datasets for evaluation
  # tfds.load returns tf.Tensors (or tf.data.Datasets if batch_size != -1)
  # You can convert them to NumPy arrays (or iterables of NumPy arrays) with tfds.dataset_as_numpy
  mnist_data, info = tfds.load(name="mnist", batch_size=-1, data_dir=data_dir, with_info=True)
  mnist_data = tfds.as_numpy(mnist_data)
  h, w, c = info.features['image'].shape

  num_pixels = h*w*c
  data = mnist_data['train']

  images, labels = data['image'], data['label']
  images = jnp.reshape(images, (len(images), num_pixels))
  

  images = images.reshape(-1,28,28)
  # subsample
  images = images[:,::2,::2]
  images = images.reshape(-1,14*14)

  nb_images = len(images)
  labels = (labels < 5).astype(np.int)
  flip = np.random.rand(nb_images) < .25
  labels = np.abs(labels - flip)
  labels = labels[:, None]

  images = images/255 

  train_len = 50000
  train_images, val_images, train_labels, val_labels = images[:train_len], images[train_len:], labels[:train_len], labels[train_len:]
  return train_images, train_labels, val_images, val_labels

def add_color(images, labels, p:float) -> Mapping[str, np.ndarray]:
  z = labels.copy()
  nb_images = len(images)
  flip = (np.random.rand(nb_images) < p).astype(int)
  flip = flip[:, None]
  z = np.squeeze(np.abs(flip - z),axis=-1)
  images = np.stack([images, images],axis=1)
  images[np.arange(nb_images),(1-z), :] = 0
  images = images.reshape(nb_images, -1)
  return {"images": images, "labels":labels}

def build_env():
  train_images, train_labels, val_images, val_labels = get_mnist()
  train_data = [
    add_color(train_images[::2], train_labels[::2], .2),
    add_color(train_images[1::2], train_labels[1::2], .1),
  ]

  test_data = [
    add_color(val_images, val_labels, .9),
  ]

  env = Environments(train_data, test_data)
  return env




def data_stream(data,  batch_size):
    images = data['images']
    labels = data['labels']
    nb_images = len(images)
    for i in range(0, nb_images, batch_size):
        yield {
            'images':images[i:i+batch_size], 
            'labels':labels[i:i+batch_size]
            }