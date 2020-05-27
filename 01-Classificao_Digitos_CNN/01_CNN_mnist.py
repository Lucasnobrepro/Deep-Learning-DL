#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 10:05:22 2020

@author: lucas
"""

# Bibliotecas
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization


# separação de dados em treinoe teste.
(X_train, y_train), (X_test, y_test) = mnist.load_data()

plt.imshow(X_train[0],cmap='gray');

# transformamos os dados para o tensorflow conseguir trabalha com os dados.
X_train = X_train.reshape(X_train.shape[0], # qtd amostras
                          28, # largura
                          28, # altura
                          1)  # dimensoes

X_test = X_test.reshape(X_test.shape[0], # qtd amostras
                          28, # largura
                          28, # altura
                          1)  # dimensoes
# Mudando o tipo.
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Mudando a escala usando min max.
X_train /= 255
X_test  /= 255

# fazemos o dummy das classes
y_train = np_utils.to_categorical(y_train,10)
y_test = np_utils.to_categorical(y_test ,10)

from numpy.random import seed
seed(1)
import tensorflow as tf
tf.random.set_seed(2)


# Criando modelo de classificador.
model = Sequential()

# Pirmeira camada: Operador de convolução
model.add(Conv2D(32,
                 (3,3),
                 input_shape=(28,28,1),
                 activation='relu'))
# nomalizando os valores da camada de convolução, reduzimos o tempo de processamento.
model.add(BatchNormalization())

# Camada de pooling
model.add(MaxPooling2D())

# Segunda camada: Operador de convolução
model.add(Conv2D(32,
                 (3,3),
                 activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())

# camada de flatten
model.add(Flatten())

# camada oculta 1
# zerando 20% das entradas
model.add(Dense(units= 128,activation = 'relu'))
model.add(Dropout(0,2))

# camada oculta 2
# zerando 20% das entradas
model.add(Dense(units= 128,activation = 'relu'))
model.add(Dropout(0,2))

# camada de saida
model.add(Dense(units=10, activation='softmax'))

# compile
model.compile(loss = 'categorical_crossentropy',
                      optimizer = 'adam', metrics = ['accuracy'])

# Treinamento
model.fit(X_train, y_train,
          batch_size = 128, epochs = 1,
          validation_split=0.3)


model.evaluate(X_test,y_test)   




