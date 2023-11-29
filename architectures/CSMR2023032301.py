#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plantilla xarxes neurals AE amb coll d'ampolla modulat 2023032301
V3.0
Sebastià Mijares i Verdú - GICI, UAB
sebastia.mijares@uab.cat

Plantilla de script per a construir, entrenar, i utilitzar xarxes neurals per a la compressió d'imatges de teledetecció hiperespectrals. Per a crear una nova xarxa, cal copiar aquest script i adaptar els paràmetres desitjats tal i com  s'explica a "com utilitzar aquest codi".

Aquest script està basat en CSMR2023011001 v6.2 i en NLTC2023022201 v1.0.

Com utilitzar aquest codi
-------------------------

Aquest codi s'executa des de un terminal. Amb la comanda -h s'hi detalla l'ajuda d'ús sobre les funcions: entrenament (train), compressió (compress), i reconstrucció (decompress). En aquesta guia s'explica  com construir models utilitzant aquesta plantilla.

Hi ha dues xarxes a entrenar: la codificadora (Analysis Transform) i la reconstructora (Synthesis transform). Per a canviar el nombre i tipus de capes de cadascuna, cal modificar en aquest script les classes AnalysisTransform i/o SynthesisTransform. Es pot trobar més informació sobre els tipus de capa a la documentació oficial de TF.

Para atenció a que els outputs de la codificadora i els inputs de la reconstructora tenen les mateixes mides (com a tensors). El nombre de filtres és el nombre de bandes (que ha de coincidir de la darrera capa de la codificadora amb la primera de la reconstructora), les strides_down divideixen per aquell valor les dimensions espacials (totes) i les strides_up multipliquen les dimensions espacials. Per tant, el producte de les strides_down de la codificadora ha de coincidir amb el producte de les strides_up de la reconstructora.

Tots els fitxers que processen aquests models són .raw, sense headers.

Paràmetres
----------

data_type: tipus de dades amb que la xarxa operarà
bit_length: nombre de bits màxims usats en cada valor

Requisits
---------

-Mòdul argparse
-Mòdul glob
-Mòdul sys
-Mòdul absl
-Mòdul tensorflow
-Mòdul tensorflow-compression

Desenvolupament
---------------

"""

import argparse
import glob
import sys
from absl import app
from absl.flags import argparse_flags
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_compression as tfc
import numpy as np
# import os

# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

"""
WARNING: THIS CODE WILL ONLY RUN FOR ONE DATATYPE (uint8, uint16, etc.).

THIS IS HARD-CODED IN THE PARAMETERS BELOW.
"""

#Parameters

data_type = tf.uint16
bit_length = 16
bitrate_precision = 0.05

#Running code

def read_raw(filename,height,width,bands,endianess, datatype):
  """
  Reads a raw image file and returns a tensor of given height, width, and number of components, taking endianess and bytes-per-entry into account.

  This function is independent from the patchsize chosen for training.
  """
  string = tf.io.read_file(filename)
  vector = tf.io.decode_raw(string, datatype, little_endian=(endianess==1))
  return tf.transpose(tf.reshape(vector, [bands, height, width]), (1,2,0))

def write_raw(filename, image):
  """
  Saves an image to a raw file.
  """
  arr = np.transpose(np.array(image),(2,0,1))
  arr.tofile(filename, format='.raw')

def get_geometry(file):
    if file[-4:]=='.raw':
        G = file.split('.')[1].split('_')
        bands = G[0]
        width = G[1]
        height = G[2]
        datatype = G[3]
        endianess = G[4]
        #Note these are all strings, not int numbers!
        return (bands, width, height, endianess, datatype)
    print('No RAW files were found. Assumes 8-bit PNG files instead.')
    return ('0', '0', '0', '0', '1')

class ModulatingTransform(tf.keras.Sequential):
  """
  The modulating or demodulating transform.
  
  The lambda input value is normalised by the maximum lambda value provided in training.
  """

  def __init__(self, hidden_nodes, num_filters, maxval):
    super().__init__()
    self.add(tf.keras.layers.Lambda(lambda x: x / maxval))
    self.add(tf.keras.layers.Dense(hidden_nodes, activation=tf.nn.relu, kernel_initializer='ones'))
    self.add(tf.keras.layers.Dense(num_filters, activation=tf.nn.relu, kernel_initializer='ones'))

class AnalysisTransform(tf.keras.Sequential):
  """The analysis transform."""

  def __init__(self, num_filters_hidden, num_filters_latent):
    super().__init__(name="analysis")
    self.add(tfc.SignalConv2D(
        num_filters_hidden, (5, 5), name="layer_0", corr=True, strides_down=2,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="gdn_0")))
    self.add(tfc.SignalConv2D(
        num_filters_hidden, (5, 5), name="layer_1", corr=True, strides_down=2,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="gdn_1")))
    self.add(tfc.SignalConv2D(
        num_filters_hidden, (5, 5), name="layer_2", corr=True, strides_down=2,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="gdn_1")))
    self.add(tfc.SignalConv2D(
        num_filters_latent, (5, 5), name="layer_3", corr=True, strides_down=2,
        padding="same_zeros", use_bias=False,
        activation=None))


class SynthesisTransform(tf.keras.Sequential):
  """The synthesis transform."""

  def __init__(self, num_filters, bands):
    super().__init__(name="synthesis")
    self.add(tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_0", corr=False, strides_up=2,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="igdn_0", inverse=True)))
    self.add(tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_1", corr=False, strides_up=2,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="igdn_1", inverse=True)))
    self.add(tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_2", corr=False, strides_up=2,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="igdn_1", inverse=True)))
    self.add(tfc.SignalConv2D(
        bands, (5, 5), name="layer_3", corr=False, strides_up=2,
        padding="same_zeros", use_bias=True,
        activation=None))

class HyperAnalysisTransform(tf.keras.Sequential):
  """The analysis transform for the entropy model parameters."""

  def __init__(self, num_filters_hidden, num_filters_latent):
    super().__init__(name="hyper_analysis")
    self.add(tfc.SignalConv2D(
        num_filters_hidden, (3, 3), name="layer_0", corr=True, strides_down=1,
        padding="same_zeros", use_bias=True,
        activation=tf.nn.relu))
    self.add(tfc.SignalConv2D(
        num_filters_hidden, (5, 5), name="layer_1", corr=True, strides_down=2,
        padding="same_zeros", use_bias=True,
        activation=tf.nn.relu))
    self.add(tfc.SignalConv2D(
        num_filters_latent, (5, 5), name="layer_2", corr=True, strides_down=2,
        padding="same_zeros", use_bias=False,
        activation=None))

class HyperSynthesisTransform(tf.keras.Sequential):
  """The synthesis transform for the entropy model parameters."""

  def __init__(self, num_filters_hidden, num_filters_latent):
    super().__init__(name="hyper_synthesis")
    self.add(tfc.SignalConv2D(
        num_filters_hidden, (5, 5), name="layer_0", corr=False, strides_up=2,
        padding="same_zeros", use_bias=True, kernel_parameter="variable",
        activation=tf.nn.relu))
    self.add(tfc.SignalConv2D(
        num_filters_hidden, (5, 5), name="layer_1", corr=False, strides_up=2,
        padding="same_zeros", use_bias=True, kernel_parameter="variable",
        activation=tf.nn.relu))
    self.add(tfc.SignalConv2D(
        num_filters_latent, (3, 3), name="layer_2", corr=False, strides_up=1,
        padding="same_zeros", use_bias=True, kernel_parameter="variable",
        activation=None))

class CSMR2023032301_v3_0(tf.keras.Model):
  """Main model class."""

  def __init__(self, lmbda, num_filters, bands, num_scales, scale_min, scale_max):
    super().__init__()
    self.bands = bands
    self.max_lambda = max(lmbda)
    self.min_lambda = min(lmbda)
    self.num_scales = num_scales
    offset = tf.math.log(scale_min)
    factor = (tf.math.log(scale_max) - tf.math.log(scale_min)) / (
        num_scales - 1.)
    if len(num_filters) == 1:
        num_filters.append(num_filters[0])
    self.scale_fn = lambda i: tf.math.exp(offset + factor * i)
    self.analysis_transform = AnalysisTransform(num_filters[0], num_filters[1])
    self.synthesis_transform = SynthesisTransform(num_filters[0], bands)
    self.hyper_analysis_transform = HyperAnalysisTransform(num_filters[0], num_filters[1])
    self.hyper_synthesis_transform = HyperSynthesisTransform(num_filters[0], num_filters[1])
    self.modulating_transform = ModulatingTransform(64, num_filters[1], max(lmbda))
    self.hyperprior = tfc.NoisyDeepFactorized(batch_shape=(num_filters[1],))
    self.build((None, None, None, bands))

  def call(self, x, training, Lmbda=None):
    """Computes rate and distortion losses."""
    entropy_model = tfc.LocationScaleIndexedEntropyModel(
        tfc.NoisyNormal, self.num_scales, self.scale_fn, coding_rank=3,
        compression=False)
    side_entropy_model = tfc.ContinuousBatchedEntropyModel(
        self.hyperprior, coding_rank=3, compression=False)
    
    m = tf.reduce_min(x, axis=(0,1,2))
    M = tf.reduce_max(x, axis=(0,1,2))
    
    lmda = tf.random.uniform((1,), minval=self.min_lambda, maxval=self.max_lambda)
    y = self.analysis_transform((x-m) / (M-m+1))
    mod = self.modulating_transform(tf.expand_dims(tf.expand_dims(tf.expand_dims(lmda,0),0),0))
    y = mod*y
    z = self.hyper_analysis_transform(abs(y))
    z_hat, side_bits = side_entropy_model(z, training=training)
    indexes = self.hyper_synthesis_transform(z_hat)
    y_hat, bits = entropy_model(y, indexes, training=training)
    demod = 1/mod
    x_hat = self.synthesis_transform(demod*y_hat)
    x_hat = (x_hat*(M-m+1))+m
    # Total number of bits divided by total number of pixels.
    num_pixels = tf.cast(tf.reduce_prod(tf.shape(x)[:-1]), bits.dtype)
    bpp = (tf.reduce_sum(bits) + tf.reduce_sum(side_bits)) / num_pixels
    # Mean squared error across pixels.
    mse = tf.reduce_mean(tf.math.squared_difference(x, x_hat))
    # The rate-distortion Lagrangian.
    loss = bpp + lmda * mse
    return loss, bpp, mse

  def train_step(self, x):
    with tf.GradientTape() as tape:
      loss, bpp, mse = self(x, training=True)
    variables = self.trainable_variables
    gradients = tape.gradient(loss, variables)
    self.optimizer.apply_gradients(zip(gradients, variables))
    self.loss.update_state(loss)
    self.bpp.update_state(bpp)
    self.mse.update_state(mse)
    return {m.name: m.result() for m in [self.loss, self.bpp, self.mse]}

  def test_step(self, x):
    loss, bpp, mse = self(x, training=False)
    self.loss.update_state(loss)
    self.bpp.update_state(bpp)
    self.mse.update_state(mse)
    return {m.name: m.result() for m in [self.loss, self.bpp, self.mse]}

  def predict_step(self, x):
    raise NotImplementedError("Prediction API is not supported.")

  def compile(self, **kwargs):
    super().compile(
        loss=None,
        metrics=None,
        loss_weights=None,
        weighted_metrics=None,
        **kwargs,
    )
    self.loss = tf.keras.metrics.Mean(name="loss")
    self.bpp = tf.keras.metrics.Mean(name="bpp")
    self.mse = tf.keras.metrics.Mean(name="mse")

  def fit(self, *args, **kwargs):
    retval = super().fit(*args, **kwargs)
    # After training, fix range coding tables.
    self.entropy_model = tfc.LocationScaleIndexedEntropyModel(
        tfc.NoisyNormal, self.num_scales, self.scale_fn, coding_rank=3,
        compression=True)
    self.side_entropy_model = tfc.ContinuousBatchedEntropyModel(
        self.hyperprior, coding_rank=3, compression=True)
    return retval

  @tf.function(input_signature=[
      tf.TensorSpec(shape=(None, None, None), dtype=data_type)
  ])
  def main_transform(self, x):
    """Applies analysis transform and returns y vector."""
    # Add batch dimension and cast to float.
    x = tf.cast(x, dtype=tf.float32)
    m = tf.reduce_min(x, axis=(0,1))
    M = tf.reduce_max(x, axis=(0,1))
    x = tf.expand_dims(x, 0)
    y = self.analysis_transform((x-m)/(M-m+1))
    x_shape = tf.shape(x)[1:-1]
    y_shape = tf.shape(y)[1:-1]
    return y, x_shape, y_shape, tf.as_string([tf.squeeze(m)]), tf.as_string([tf.squeeze(M)])

  @tf.function(input_signature=[
      tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32),
      tf.TensorSpec(shape=(2,), dtype=tf.int32),
      tf.TensorSpec(shape=(2,), dtype=tf.int32),
      tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32)
  ])
  def modulation(self, y, x_shape, y_shape, lmda):
    """Applies modulation transform, encodes latent representation and side information, and returns strings."""
    
    lmda_send = tf.round(255*(lmda-self.min_lambda)/(self.max_lambda-self.min_lambda))
    mod = self.modulating_transform(lmda_send)
    y = mod*y
    z = self.hyper_analysis_transform(abs(y))
    z_shape = tf.shape(z)[1:-1]
    z_hat, _ = self.side_entropy_model(z, training=False)
    indexes = self.hyper_synthesis_transform(z_hat)
    indexes = indexes[:, :y_shape[0], :y_shape[1], :]
    side_string = self.side_entropy_model.compress(z)
    string = self.entropy_model.compress(y, indexes)
    bps = tf.cast((tf.strings.length(string)+tf.strings.length(side_string))*8/(tf.reduce_prod(x_shape)*self.bands), dtype=tf.float32)
    return string, side_string, z_shape, tf.cast(lmda_send, tf.int8), bps
                    
  @tf.function(input_signature=[
      tf.TensorSpec(shape=(1,), dtype=tf.string),
      tf.TensorSpec(shape=(1,), dtype=tf.string),
      tf.TensorSpec(shape=(2,), dtype=tf.int32),
      tf.TensorSpec(shape=(2,), dtype=tf.int32),
      tf.TensorSpec(shape=(2,), dtype=tf.int32),
      tf.TensorSpec(shape=(None, None, None, None), dtype=tf.int8),
      tf.TensorSpec(shape=(None,), dtype=tf.string),
      tf.TensorSpec(shape=(None,), dtype=tf.string)
  ])
  def decompress(self, string, side_string, x_shape, y_shape, z_shape, L, m_string, M_string):
    """Decompresses an image."""
    V = (tf.cast(L, tf.float32)/255.)*(self.max_lambda-self.min_lambda)+self.min_lambda
    m = tf.strings.to_number(m_string)
    M = tf.strings.to_number(M_string)
    demod = 1/self.modulating_transform(V)
    z_hat = self.side_entropy_model.decompress(side_string, z_shape)
    indexes = self.hyper_synthesis_transform(z_hat)
    indexes = indexes[:, :y_shape[0], :y_shape[1], :]
    y_hat = self.entropy_model.decompress(string, indexes)
    x_hat = (self.synthesis_transform(demod*y_hat)*(M-m+1))+m
    # Remove batch dimension, and crop away any extraneous padding.
    x_hat = x_hat[0, :x_shape[0], :x_shape[1], :]
    # Then cast back to data type.
    return tf.saturate_cast(tf.saturate_cast(tf.round(x_hat), tf.int32), data_type)
  
  @tf.function(input_signature=[
      tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32)
  ])
  def plain_modulation(self, lmda):
    """Applies analysis transform and returns y vector."""
    # Add batch dimension and cast to float.
    mod = self.modulating_transform(lmda)
    return mod

  @tf.function(input_signature=[
      tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32),
      tf.TensorSpec(shape=(None,), dtype=tf.float32),
      tf.TensorSpec(shape=(None,), dtype=tf.float32)
  ])
  def inverse_transform(self, y, m, M):
    """Decompresses an image."""
    x_hat = (self.synthesis_transform(y)*(M-m+1))+m
    # Remove batch dimension, and crop away any extraneous padding.
    # Then cast back to data type.
    return tf.saturate_cast(tf.saturate_cast(tf.round(x_hat), tf.int32), data_type)


def check_image_size(image, patchsize,bands):
  shape = tf.shape(image)
  return shape[0] >= patchsize and shape[1] >= patchsize and shape[-1] == bands


def crop_image(image, patchsize,bands):
  image = tf.image.random_crop(image, (patchsize, patchsize, bands))
  return tf.cast(image, tf.float32)


def get_dataset(name, split, args):
  """Creates input data pipeline from a TF Datasets dataset."""
  with tf.device("/cpu:0"):
    dataset = tfds.load(name, split=split, shuffle_files=True)
    if split == "train":
      dataset = dataset.repeat()
    dataset = dataset.filter(
        lambda x: check_image_size(x["image"], args.patchsize, args.bands))
    dataset = dataset.map(
        lambda x: crop_image(x["image"], args.patchsize, args.bands))
    dataset = dataset.batch(args.batchsize, drop_remainder=True)
  return dataset


def get_custom_dataset(split, args):
  """Creates input data pipeline from custom raw images."""
  with tf.device("/cpu:0"):
    files = glob.glob(args.train_glob)
    if not files:
      raise RuntimeError(f"No training images found with glob "
                         f"'{args.train_glob}'.")
    dataset = tf.data.Dataset.from_tensor_slices(files)
    dataset = dataset.shuffle(len(files), reshuffle_each_iteration=True)
    if split == "train":
      dataset = dataset.repeat()
    dataset = dataset.map(
        lambda x: crop_image(read_raw(x, args.height, args.width, args.bands, args.endianess, data_type), args.patchsize, args.bands),
        num_parallel_calls=args.preprocess_threads)
    dataset = dataset.batch(args.batchsize, drop_remainder=True)
  return dataset


def train(args):
  """Instantiates and trains the model."""
  if args.check_numerics:
    tf.debugging.enable_check_numerics()
  model = CSMR2023032301_v3_0(args.lmbda, args.num_filters, args.bands, args.num_scales, args.scale_min, args.scale_max)
  model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
  )

  if args.train_glob:
    train_dataset = get_custom_dataset("train", args)
    validation_dataset = get_custom_dataset("validation", args)
  else:
    train_dataset = get_dataset("clic", "train", args)
    validation_dataset = get_dataset("clic", "validation", args)
  validation_dataset = validation_dataset.take(args.max_validation_steps)

  model.fit(
      train_dataset.prefetch(8),
      epochs=args.epochs,
      steps_per_epoch=args.steps_per_epoch,
      validation_data=validation_dataset.cache(),
      validation_freq=1,
      callbacks=[
          tf.keras.callbacks.TerminateOnNaN(),
          tf.keras.callbacks.TensorBoard(
              log_dir=args.train_path,
              histogram_freq=1, update_freq="epoch"),
          tf.keras.callbacks.experimental.BackupAndRestore(args.train_path),
      ],
      verbose=int(args.verbose),
  )
  model.save(args.model_path)


def compress(args):
  """Compresses an image."""
  # Load model and use it to compress the image.
  model = tf.keras.models.load_model(args.model_path)
  inputs = glob.glob(args.input_file)
  for input_file in inputs:
      if args.height == None or args.width == None or args.bands == None:
          bands, width, height, endianess, datatype = get_geometry(input_file.split('/')[-1])
          x = read_raw(input_file, int(height), int(width), int(bands), int(endianess), data_type)
      else:
          x = read_raw(input_file, args.height, args.width, args.bands, args.endianess, data_type)
      y, x_shape, y_shape, m_string, M_string = model.main_transform(x)
      if tf.rank(m_string) == 2:
          m_string = tf.squeeze(m_string)
          M_string = tf.squeeze(M_string)
      
      if args.quality_array:
          # Note the array is expected to be stored with the same endianness as the main array.
          Q = tf.expand_dims(read_raw(args.quality_array, int(height)//16, int(width)//16, 1, int(endianess), tf.float32),0)
      else:
          Q = tf.ones((1, 1, 1, 1))*args.quality
      
      string_max, side_string_max, z_shape, L_max, bps_max = model.modulation(y, x_shape, y_shape, Q)
      
      if args.bitrate == None:
          string, side_string, L = string_max, side_string_max, L_max
      else:
          string_min, side_string_min, z_shape, L_min, bps_min = model.modulation(y, x_shape, y_shape, tf.constant([0], dtype=tf.float32))
          max_lmbda = args.quality
          min_lmbda = 0
          iteration = 1
          while iteration < 101: #limited to 100 iterations to avoid infinite loops.
            iteration += 1
            if args.bitrate > bps_max-bitrate_precision or iteration == 100: #limited to 100 iterations to avoid infinite loops.
                string, side_string, L = string_max, side_string_max, L_max
                break
            elif args.bitrate < bps_min+bitrate_precision:
                string, side_string, L = string_min, side_string_min, L_min
                break
            mid_lmbda = (max_lmbda+min_lmbda)/2
            string_mid, side_string_mid, z_shape, L_mid, bps_mid = model.modulation(y, x_shape, y_shape, tf.constant([mid_lmbda]))        
            if bps_mid > args.bitrate:
                string_max, side_string_max, L_max, max_lmbda, bps_max = string_mid, side_string_mid, L_mid, mid_lmbda, bps_mid
            else:
                min_lmbda = mid_lmbda
                string_min, side_string_min, L_min, min_lmbda, bps_min = string_mid, side_string_mid, L_mid, mid_lmbda, bps_mid
      
      L = tf.reshape(L, (-1,))
      tensors = string, side_string, x_shape, y_shape, z_shape, L, m_string, M_string
    
      # Write a binary file with the shape information and the compressed string.
      packed = tfc.PackedTensors()
      packed.pack(tensors)
      with open(input_file+'.tfci', "wb") as f:
        f.write(packed.string)

def transform(args):
  """Compresses an image."""
  # Load model and use it to compress the image.
  model = tf.keras.models.load_model(args.model_path)
  inputs = glob.glob(args.input_file)
  for input_file in inputs:
      if args.height == None or args.width == None or args.bands == None:
          bands, width, height, endianess, datatype = get_geometry(input_file.split('/')[-1])
          geometry = '.'+input_file.split('/')[-1].split('.')[-2]+'.raw'
          path = input_file.replace(geometry,'')
          x = read_raw(input_file, int(height), int(width), int(bands), int(endianess), data_type)
      else:
          x = read_raw(input_file, args.height, args.width, args.bands, args.endianess, data_type)
          path = input_file
          
      y, x_shape, y_shape, m_string, M_string = model.main_transform(x)
      
      print(m_string)
      print(M_string)
          
      write_raw(path+'_noquant.192_'+str(int(width)//16)+'_'+str(int(height)//16)+'_6_1_0.raw', y[0,:,:,:])

def modulation(args):
  """Compresses an image."""
  # Load model and use it to compress the image.
  model = tf.keras.models.load_model(args.model_path)
  mod = model.plain_modulation(tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.constant([args.quality]), 0), 0), 0))
  write_raw('./modulated_'+str(args.quality)+'.192_1_1_6_1_0.raw', mod[0,:,:,:])

def inverse_transform(args):
  """Compresses an image."""
  # Load model and use it to compress the image.
  model = tf.keras.models.load_model(args.model_path)
  inputs = glob.glob(args.input_file)
  for input_file in inputs:
      if args.height == None or args.width == None or args.bands == None:
          bands, width, height, endianess, datatype = get_geometry(input_file.split('/')[-1])
          geometry = '.'+input_file.split('/')[-1].split('.')[-2]+'.raw'
          path = input_file.replace(geometry,'')
          if datatype == '0':
            D = np.bool_
          elif datatype == '1':
            D = tf.uint8
          elif datatype == '2':
            D = tf.uint16
          elif datatype == '3':
            D = tf.int16
          elif datatype == '4':
            D = tf.int32
          elif datatype == '5':
            D = tf.int64
          elif datatype == '6':
            D = tf.float32
          else:
            D = tf.float64
          
          y = read_raw(input_file, int(height), int(width), int(bands), int(endianess), D)
      else:
          y = read_raw(input_file, args.height, args.width, args.bands, args.endianess, tf.float32)
          path = input_file
          bands, width, height, endianess = args.bands, args.width, args.height, args.endianess
      
      x_hat = model.inverse_transform(tf.expand_dims(y, 0), tf.constant([args.minimum]), tf.constant([args.maximum]))
      
      write_raw(path+'.1_'+str(int(width)*16)+'_'+str(int(height)*16)+'_2_1_0.raw.tfci.raw', x_hat[0,:,:,:])


def decompress(args):
  """Decompresses an image."""
  # Load the model and determine the dtypes of tensors required to decompress.
  model = tf.keras.models.load_model(args.model_path)
  dtypes = [t.dtype for t in model.decompress.input_signature]
  inputs = glob.glob(args.input_file)
  for input_file in inputs:
      # Read the shape information and compressed string from the binary file,
      # and decompress the image using the model.
      with open(input_file, "rb") as f:
        packed = tfc.PackedTensors(f.read())
      string, side_string, x_shape, y_shape, z_shape, L, m_string, M_string = packed.unpack(dtypes)
      try:
          L = tf.reshape(L, (1, y_shape[0], y_shape[1], 1))
      except:
          L = tf.reshape(L, (1, 1, 1, 1))
      tensors = string, side_string, x_shape, y_shape, z_shape, L, m_string, M_string
      x_hat = model.decompress(*tensors)
    
      # Write reconstructed image out as a RAW file.
      write_raw(input_file+'.raw', x_hat)


def parse_args(argv):
  """Parses command line arguments."""
  parser = argparse_flags.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  # High-level options.
  parser.add_argument(
      "--verbose", "-V", action="store_true",
      help="Report progress and metrics when training or compressing.")
  parser.add_argument(
      "--model_path", default="CSMR2023032301_v3_0",
      help="Path where to save/load the trained model.")
  subparsers = parser.add_subparsers(
      title="commands", dest="command",
      help="What to do: 'train' loads training data and trains (or continues "
           "to train) a new model. 'compress' reads an image file (lossless "
           "raw format) and writes a compressed binary file. 'decompress' "
           "reads a binary file and reconstructs the image (in raw format). "
           "input and output filenames need to be provided for the latter "
           "two options. Invoke '<command> -h' for more information.")

  # 'train' subcommand.
  train_cmd = subparsers.add_parser(
      "train",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Trains (or continues to train) a new model. Note that this "
                  "model trains on a continuous stream of patches drawn from "
                  "the training image dataset. An epoch is always defined as "
                  "the same number of batches given by --steps_per_epoch. "
                  "The purpose of validation is mostly to evaluate the "
                  "rate-distortion performance of the model using actual "
                  "quantization rather than the differentiable proxy loss. "
                  "Note that when using custom training images, the validation "
                  "set is simply a random sampling of patches from the "
                  "training set.")
  train_cmd.add_argument(
      "--lambda", type=float, nargs='+', default=[0.01], dest="lmbda",
      help="Lambda for rate-distortion tradeoff. Lambda values in training will be uniformely sampled in the interval between the minimum and maximum lambda values provided.")
  train_cmd.add_argument(
      "--train_glob", type=str, default=None,
      help="Glob pattern identifying custom training data. This pattern must "
           "expand to a list of images in raw format. If unspecified, the "
           "CLIC dataset from TensorFlow Datasets is used.")
  train_cmd.add_argument(
      "--num_filters", type=int, nargs='+', default=[64,192],
      help="Number of filters per layer. The first input will be the number of "
      "filters in the hidden layers and the second input in the latent layers. "
      "Later inputs will be ignored (only for backwards compatibility).")
  train_cmd.add_argument(
      "--num_filters_1D", type=int, default=4,
      help="Number of filters in the 1D layer.")
  train_cmd.add_argument(
      "--num_scales", type=int, default=64,
      help="Number of Gaussian scales to prepare range coding tables for.")
  train_cmd.add_argument(
      "--scale_min", type=float, default=.11,
      help="Minimum value of standard deviation of Gaussians.")
  train_cmd.add_argument(
      "--scale_max", type=float, default=256.,
      help="Maximum value of standard deviation of Gaussians.")
  train_cmd.add_argument(
      "--train_path", default="/tmp/CSMR2023032301_v3_0",
      help="Path where to log training metrics for TensorBoard and back up "
           "intermediate model checkpoints.")
  train_cmd.add_argument(
      "--batchsize", type=int, default=8,
      help="Batch size for training and validation.")
  train_cmd.add_argument(
      "--patchsize", type=int, default=256,
      help="Size of image patches for training and validation.")
  train_cmd.add_argument(
      "--epochs", type=int, default=1000,
      help="Train up to this number of epochs. (One epoch is here defined as "
           "the number of steps given by --steps_per_epoch, not iterations "
           "over the full training dataset.)")
  train_cmd.add_argument(
      "--steps_per_epoch", type=int, default=1000,
      help="Perform validation and produce logs after this many batches.")
  train_cmd.add_argument(
      "--max_validation_steps", type=int, default=16,
      help="Maximum number of batches to use for validation. If -1, use one "
           "patch from each image in the training set.")
  train_cmd.add_argument(
      "--preprocess_threads", type=int, default=16,
      help="Number of CPU threads to use for parallel decoding of training "
           "images.")
  train_cmd.add_argument(
      "--check_numerics", action="store_true",
      help="Enable TF support for catching NaN and Inf in tensors.")
  train_cmd.add_argument(
      "--bands", type=int, default=3, dest="bands",
      help="Number of bands in the images to train the model.")
  train_cmd.add_argument(
      "--width", type=int, default=256, dest="width",
      help="Width of the images to train the model. All must be the same size.")
  train_cmd.add_argument(
      "--height", type=int, default=256, dest="height",
      help="Height of the images to train the model. All must be the same size.")
  train_cmd.add_argument(
      "--endianess", type=int, default=0, dest="endianess",
      help="Set to 0 if data is big endian, 1 if it's little endian.")
  train_cmd.add_argument(
      "--learning_rate", type=float, default=1e-4, dest="learning_rate",
      help="Learning rate for the training session.")

  # 'compress' subcommand.
  compress_cmd = subparsers.add_parser(
      "compress",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Reads a raw file, compresses it, and writes a TFCI file.")

  # 'decompress' subcommand.
  decompress_cmd = subparsers.add_parser(
      "decompress",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Reads a TFCI file, reconstructs the image, and writes back "
                  "a raw file.")
  
  # 'transform' subcommand.
  transform_cmd = subparsers.add_parser(
      "transform",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Reads a raw file, applies the main transform on it, and writes a raw file.")
  
  # 'inverse transform' subcommand.
  inv_transform_cmd = subparsers.add_parser(
      "inverse_transform",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Reads a raw file float32, applies the decoder main transform on it, and writes a raw file.")
  
  # 'transform' subcommand.
  modulation_cmd = subparsers.add_parser(
      "modulation",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Given a quality parameter (lambda), constructs the modulation vector and stores it as a raw file.")


  compress_cmd.add_argument("--quality",
                            type=float, default=0.1,
                            help="Lambda value regulating the rate-distortion tradeoff. A higher lambda should yield higher quality and rate images. If --bitrate is used, this lambda value will be used as the maximum lambda value in binary search."
                            )
  compress_cmd.add_argument("--quality_array",
                            type=str,
                            help="Array with lambda values regulating the rate-distortion tradeoff for each latent pixel."
                            )
  compress_cmd.add_argument("--bitrate",
                            type=float, default=None,
                            help="Bitrate to compress the images at, in bits per sample (bps, equivalent to bits per pixel per band)."
                            )
  
  inv_transform_cmd.add_argument("--minimum",
                            type=float, default=None,
                            help="Minimum value in the original image for denormalisation."
                            )
  
  inv_transform_cmd.add_argument("--maximum",
                            type=float, default=None,
                            help="Maximum value in the original image for denormalisation."
                            )
  
  modulation_cmd.add_argument("quality",
                            type=float, default=0.0001,
                            help="Lambda value regulating the rate-distortion tradeoff."
                            )
  
  # Arguments for both 'compress' and 'decompress'.
  for cmd, ext in ((compress_cmd, ".tfci"), (decompress_cmd, ".raw"), (transform_cmd, ".tfci"), (inv_transform_cmd, ".raw")):
    cmd.add_argument(
        "input_file",
        help='Input filename or glob pattern. If a glob pattern is used, delimitate it with "".')
    cmd.add_argument(
        "--bands",type=int,default=None,
        help="Number of bands in input image.")
    cmd.add_argument(
        "--width",type=int,default=None,
        help="Input image width.")
    cmd.add_argument(
        "--height",type=int,default=None,
        help="Input image height.")
    cmd.add_argument(
        "--endianess",type=int,default=None,
        help="Set to 0 if data is big endian, 1 if it's little endian.")

  # Parse arguments.
  args = parser.parse_args(argv[1:])
  if args.command is None:
    parser.print_usage()
    sys.exit(2)
  return args


def main(args):
  # Invoke subcommand.
  if args.command == "train":
    train(args)
  elif args.command == "compress":
    compress(args)
  elif args.command == "transform":
    transform(args)
  elif args.command == "inverse_transform":
    inverse_transform(args)
  elif args.command == "decompress":
    decompress(args)
  elif args.command == "modulation":
    modulation(args)


if __name__ == "__main__":
  app.run(main, flags_parser=parse_args)
