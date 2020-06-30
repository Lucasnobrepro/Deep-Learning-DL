# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 18:34:48 2020

@author: Lucas Nobre
"""

# Importando Bibliotecas
from keras.models import Sequential
from keras.layers import MaxPooling2D, Flatten, Conv2D, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

# Criando modelo
model = Sequential()

# Primeira camada de convolução
model.add(Conv2D(32,
                 (3,3),
                 input_shape=(32,32,3),
                 activation='relu'))

# acelerando o processamento atrvés da normalização dos dados.
model.add(BatchNormalization)

# Camada de pooling.
model.add(MaxPooling2D(pool_size = (2.2)))

# Segunda camada de convolução
model.add(Conv2D(32,
                 (3,3),
                 input_shape=(32,32,3),
                 activation='relu'))

# acelerando o processamento atrvés da normalização dos dados.
model.add(BatchNormalization)

# Camada de pooling.
model.add(MaxPooling2D(pool_size = (2.2)))

# Camada de flatten
model.add(Flatten())

#Criando uma rede neural densa.
model.add(Dense(units = 128,
                activation='relu'))
# zerando 20% das entradas para evitar overfitting
model.add(Dropout(0.2))

#Criando uma rede neural densa.
model.add(Dense(units = 128,
                activation='relu'))
# zerando 20% das entradas para evitar overfitting
model.add(Dropout(0.2))

# saida
model.add(Dense(units = 1,
                activation='sigmoid'))


model.compile(optimize='Adam',
              loss='binary_crossentropy',
              metrics='accuracy')





















