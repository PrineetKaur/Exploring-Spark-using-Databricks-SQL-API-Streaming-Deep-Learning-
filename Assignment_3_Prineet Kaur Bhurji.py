# Databricks notebook source
"""
BDT2 - Assignment 3 - Deep Learning
"""

#This assignment is a guided notebook for Assignment 3 on Deep Learning.You can already complete the exercises in this notebook after the first session on Deep Learning.This exercise introduces single-label multiclass classification using Deep Learning. This is a popular technique for f.ex. image classification.

# COMMAND ----------

pip install tensorflow --ignore-installed

# COMMAND ----------

from tensorflow.keras.datasets import reuters
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
#The dataset used here contains Reuters newswires that should be classified into 46 mutually exclusive topics.

#x
import numpy as np

def vectorize_sequences(sequences, dimension=10000):
  results = np.zeros((len(sequences), dimension))
  for i, sequence in enumerate(sequences):
    results[i, sequence] = 1.
  return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

#y
from tensorflow.keras.utils import to_categorical

y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)

#Alternative: encode the labels by casting them to an integer tensor
#y_train = np.array(train_labels)
#y_test = np.array(test_labels)
#Remark: in this case, you need to choose sparse_categorical_crossentropy as the loss function 
#since categorical_crossentropy expects the labels to follow a categorical encoding

#Split the training dataset in a training and validation set
x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = y_train[:1000]
partial_y_train = y_train[1000:]

partial_y_train.shape

# COMMAND ----------

#Exercise 1A: Estimate a model with 3 layers and 1 output layer.
#Determine the number of hidden units as follows:
  #Use the same number of units for the first 3 layers.
  #For these first 3 layers, use a number of units that is larger than the number of output classes but smaller than 100 (taking into account the 2^n rule).

# COMMAND ----------

#Estimate a DL model with 2 layers of 16 units and 1 output layer.
from tensorflow.keras import models
from tensorflow.keras import layers

#Define the model
model = models.Sequential()

#Define the layers
model.add(layers.Dense(64, activation='relu', input_shape=(partial_x_train.shape[1],)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='sigmoid'))

#Define an optimizer, loss function, and metric for success.

model.compile(optimizer='rmsprop',
  loss='binary_crossentropy',
  metrics=['acc'])

history = model.fit(partial_x_train, 
          partial_y_train, 
          epochs=20, 
          batch_size=512,
          validation_data=(x_val, y_val))

#Fit the model. Use batch_size=512 and epoch=20 as starting values.

#Get accuracy
history_dict = history.history
acc = history_dict.get("acc")
val_acc = history_dict.get("val_acc")
print(val_acc)

#Look at the results (accuracy)
import matplotlib.pyplot as plt
plt.clf()

#Create values
acc_values = history.history['acc']
val_acc_values = history.history['val_acc']
epochs = range(1, len(acc_values) + 1)

#Create plot
plt.plot(epochs, acc_values, 'bo', label='Training acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#At which value for epoch does the network overfit?
#Copy-paste the answer in the comments.

#The network overfits around 7.5 epochs, as the validation curve goes under the
#training curve.

#Retrain the model with the optimal epoch
model.compile(optimizer='rmsprop',
  loss='binary_crossentropy',
  metrics=['acc'])

history2 = model.fit(partial_x_train, partial_y_train, epochs=5, batch_size=512)

results_epoch_optim = model.evaluate(x_test, y_test)

#Retrain the model with the optimal number of epochs. What is the final accuracy?
#Copy-paste the answer in the comments.

"""The final accuracy is 0.9580 after going through all of the layers."""

# COMMAND ----------

#Exercise 1B: Fit the model in Exercise 1 again but with a batch_size=32. What can you conclude in terms of accuracy?
#Copy-paste the accuracy and your answer in the comments.

# COMMAND ----------

history = model.fit(partial_x_train, 
          partial_y_train, 
          epochs=20, 
          batch_size=32,
          validation_data=(x_val, y_val))

#Get accuracy
history_dict = history.history
acc = history_dict.get("acc")
val_acc = history_dict.get("val_acc")
print(val_acc)

"""
The accuracy goes up to 0.9598, which means that reducing the batch_size to 32 has had considerable impact on our model.
"""

# COMMAND ----------

#Exercise 2: Starting from the model in Exercise 1, estimate a model where the 1st layer is doubled in terms of units.
#Add an additional layer after the first layer where the number of units is also doubled.
#What can you conclude on the accuracy of the model compared to the model in Exercise 1?
#Copy-paste the accuracy of this model and your conclusion in the comments.

# COMMAND ----------

from tensorflow.keras import models
from tensorflow.keras import layers

#Define the model
model = models.Sequential()

#Define the layers
model.add(layers.Dense(128, activation='relu', input_shape=(partial_x_train.shape[1],)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='sigmoid'))

model.compile(optimizer='rmsprop',
  loss='binary_crossentropy',
  metrics=['acc'])

history = model.fit(partial_x_train, 
          partial_y_train, 
          epochs=20, 
          batch_size=32,
          validation_data=(x_val, y_val))

"""
The final accuracy is 0.9634 which is higher than the one at the end of the first exercise. Therefore, we can say that adding a layer to the model has increased the prediction accuracy for the model.
"""

# COMMAND ----------

#Exercise 3: Starting from the model in Exercise 2, estimate a model where you add an additional layer of 64 units after the 2nd layer. 
#What can you conclude about the accuracy of the model compared to the model in Exercise 2?
#Copy-paste the accuracy of this model and your conclusion in the comments.

# COMMAND ----------

from tensorflow.keras import models
from tensorflow.keras import layers

#Defining the Model
model = models.Sequential()

#Defining the Layers
model.add(layers.Dense(128, activation='relu', input_shape=(partial_x_train.shape[1],)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='sigmoid'))

model.compile(optimizer='rmsprop',
  loss='binary_crossentropy',
  metrics=['acc'])

history = model.fit(partial_x_train, 
          partial_y_train, 
          epochs=20, 
          batch_size=32,
          validation_data=(x_val, y_val))

"""
The final accuracy for this version of the model is 0.9639, which is a little bit lower than
"""

# COMMAND ----------

#Exercise 4: Starting from the model in Exercise 2, change the number of units to 4 for the 2nd layer.
#What can you conclude about the accuracy of the model compared to the model in Exercise 2?
#Copy-paste the accuracy of this model and your conclusion in the comments.

# COMMAND ----------

from tensorflow.keras import models
from tensorflow.keras import layers

#Defining the Model
model = models.Sequential()

#Defining the Layers
model.add(layers.Dense(128, activation='relu', input_shape=(partial_x_train.shape[1],)))
model.add(layers.Dense(4, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='sigmoid'))

model.compile(optimizer='rmsprop',
  loss='binary_crossentropy',
  metrics=['acc'])

history = model.fit(partial_x_train, 
          partial_y_train, 
          epochs=20, 
          batch_size=32,
          validation_data=(x_val, y_val))

"""
Changing the number of layers in the second to 4 reduce the final accuracy to 0.9149. Therefore, we conclude that the number of units in the successive layers has reduced the prediction accuracy.
"""
