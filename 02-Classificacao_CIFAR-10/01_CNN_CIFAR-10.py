#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 11:00:33 2020

@author: lucas
"""

# Bibliotecas
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import  StratifiedKFold 
import numpy as np

seed = 5
np.random.seed(seed)

# PRE-PROCESSAMENTO

# separação de dados em treinoe teste.
(X, y), (X_test, y_test) = cifar10.load_data()

# transformamos os dados para o tensorflow conseguir trabalha com os dados.
k = X.reshape(X.shape[0], # qtd amostras
            32, # largura
            32, # altura
            3)  # dimensoes

# Mudando o tipo.
X = X.astype('float32')

# Mudando a escala usando min max.
X /= 255

# fazemos o dummy das classes
y = np_utils.to_categorical(y,10)

# Validacao cruzada
Kfold = StratifiedKFold(n_splits=5, shuffle=True,random_state=seed)
score = []

for id_train, id_test in Kfold.split(X=X,
                                     y=np.zeros(shape = (y.shape[0], 1))):
    # Criando modelo de classificador.
    model = Sequential()
    
    # Pirmeira camada: Operador de convolução
    model.add(Conv2D(32,
                     (3,3),
                     input_shape=(32,32,3),
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
    
    model.fit(X[id_train], y[id_train],
                      batch_size = 128, epochs = 5)
    
    precisao = model.evaluate(X[id_test], y[id_test])
    score.append(precisao[1])
    
    
media = sum(score) / len(score)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    





