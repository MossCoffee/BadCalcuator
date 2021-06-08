# Bad calcuator
#* TODO: read in text from text files
# https://www.tensorflow.org/tutorials/load_data/csv
# The pipeline for a text model might involve extracting symbols from raw text data, converting them to embedding identifiers with a lookup table, and batching together sequences of different lengths. 
# Neural net?
#* TODO: Do a flip (a neural net)
#* TODO: Output ints

#Clip: I need you to appreciate just how complicatated this 
# is for a first time user - and this documentation is
# abnormally good. 

# Conceptually neural nets are super simple. 
# This is a disaster nightmare trainwreck 

# TensorFlow and tf.keras
import tensorflow as tf 
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
tfds.disable_progress_bar()

# Helper libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])

calc_train = pd.read_csv(
    "C:/Programming/NeuralNetspt1/train.txt",
    names=["answer","equation"])

calc_train.head()
calc_equations = calc_train.copy()
calc_labels = calc_equations.pop("answer")

calc_equations = np.array(calc_equations)

calc_model = tf.keras.Sequential([
  layers.Dense(64),
  layers.Dense(1)
])

calc_model.compile(loss = tf.losses.MeanSquaredError(),
                      optimizer = tf.optimizers.Adam())

calc_model.fit(calc_equations, calc_labels, epochs=10)