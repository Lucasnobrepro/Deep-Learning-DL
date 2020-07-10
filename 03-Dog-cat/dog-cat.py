# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 18:34:48 2020

@author: Lucas Nobre
"""
from keras.models import Sequential
from keras.layers import MaxPooling2D, Flatten, Conv2D, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization
from keras.preprocessing import image
import  numpy as np

# Criando modelo
model = Sequential()

# Primeira camada de convolução
model.add(Conv2D(32,
                 (3,3),
                 input_shape=(64,64,3),
                 activation='relu'))

# acelerando o processamento atrvés da normalização dos dados.
model.add(BatchNormalization())

# Camada de pooling.
model.add(MaxPooling2D(pool_size = (2,2)))

# Segunda camada de convolução
model.add(Conv2D(32,
                 (3,3),
                 input_shape=(64,64,3),
                 activation='relu'))

# acelerando o processamento atrvés da normalização dos dados.
model.add(BatchNormalization())

# Camada de pooling.
model.add(MaxPooling2D(pool_size = (2,2)))

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

model.compile(optimizer='Adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
    
# argumentetion e normalizacao
train_generator = ImageDataGenerator(rescale=1./255,         # Normalizacao
                                     rotation_range=0.7,     # grau de rotacao
                                     horizontal_flip=True,   # giro horizontal
                                     shear_range=0.2,        # direcao dos pixels
                                     height_shift_range=0.07,# faixa de mudanca da altura
                                     zoom_range=0.2 )        # zoom
# normalizacao das imagens de teste.
test_generator = ImageDataGenerator(rescale=1./255)


path_train = '/home/lucas/Documentos/Git-Repositorios/Deep-Learning-DL/03-Dog-cat/dataset/training_set'
base_train = train_generator.flow_from_directory(path_train,    
                                                 target_size=(64,64),
                                                 batch_size=32,
                                                 class_mode='binary')

path_test = '/home/lucas/Documentos/Git-Repositorios/Deep-Learning-DL/03-Dog-cat/dataset/test_set'
base_test = test_generator.flow_from_directory(path_test,    
                                                 target_size=(64,64),
                                                 batch_size=32,
                                                 class_mode='binary')

# treinando rede neural
model.fit_generator(base_train, steps_per_epoch=4000/32,
                    epochs=5, validation_data=base_test,
                    validation_steps=1000/32)

# TESTANDO COM UMA UNICA IMAGEM
# lendo uma imagem
image_test = image.load_img('dataset/test_set/gato/cat.3500.jpg',
                            target_size=(64,64))

# mudando fomato da image.
image_test = image.img_to_array(image_test)

# normalizando a imagem
image_test /= 255

image_test = np.expand_dims(image_test,axis=0)

pred = model.predict(image_test)

base_train.class_indices





