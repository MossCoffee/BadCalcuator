# Bad calcuator
import collections
import pathlib
import re
import string
import os
# TensorFlow and tf.keras
import tensorflow as tf 
from tensorflow.keras import layers
from tensorflow.keras import preprocessing
from tensorflow.keras import utils
from tensorflow.keras import losses
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


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

base_dir = 'F:/'
train_dir = base_dir + 'train'

batch_size = 32
seed = 42

#loading data
raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
  train_dir,
  batch_size=batch_size,
  seed=seed)

val_dir = base_dir + 'validate'
raw_val_ds = preprocessing.text_dataset_from_directory(
    val_dir, batch_size=batch_size)

test_dir = base_dir + 'test'
raw_test_ds = preprocessing.text_dataset_from_directory(
    test_dir, batch_size=batch_size)


#Setting up the text processing
VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 250

int_vectorize_layer = TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=MAX_SEQUENCE_LENGTH)


# Make a text-only dataset (without labels), then call adapt
train_text = raw_train_ds.map(lambda text, labels: text)
int_vectorize_layer.adapt(train_text)



def binary_vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return binary_vectorize_layer(text), label

def int_vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return int_vectorize_layer(text), label


int_train_ds = raw_train_ds.map(int_vectorize_text)
int_val_ds = raw_val_ds.map(int_vectorize_text)
int_test_ds = raw_test_ds.map(int_vectorize_text)


#configuring for perf
AUTOTUNE = tf.data.AUTOTUNE

def configure_dataset(dataset):
  return dataset.cache().prefetch(buffer_size=AUTOTUNE)

int_train_ds = configure_dataset(int_train_ds)
int_val_ds = configure_dataset(int_val_ds)
int_test_ds = configure_dataset(int_test_ds)

#the actual training - who knows what this actually does

def create_model(vocab_size, num_labels):
  model = tf.keras.Sequential([
      layers.Embedding(vocab_size, 64, mask_zero=True),
      layers.Conv1D(64, 5, padding="valid", activation="relu", strides=2),
      layers.GlobalMaxPooling1D(),
      layers.Dense(num_labels)
  ])
  return model

# vocab_size is VOCAB_SIZE + 1 since 0 is used additionally for padding.
int_model = create_model(vocab_size=VOCAB_SIZE + 1, num_labels=10)

checkpoint_path = "BadCalc/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

int_model.compile(
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy'],
    callbacks=[cp_callback])
history = int_model.fit(int_train_ds, validation_data=int_val_ds, epochs=5)

int_loss, int_accuracy = int_model.evaluate(int_test_ds)

print("Int model accuracy: {:2.2%}".format(int_accuracy))

#Reload Code
# Create a basic model instance
model = create_model()

# Loads the weights
model.load_weights(checkpoint_path)

# Re-evaluate the model
loss, acc = model.evaluate(int_test_ds)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
#save model
