#! python

# IA 2020 II PAC, Tarea.
# Red neuronal con Keras para la clasificacion de XOR

# Tutorial de referencia - Aprende Machine Learning
# https://www.aprendemachinelearning.com/una-sencilla-red-neuronal-en-python-con-keras-y-tensorflow/

# Made by MilanDroid
# https://github.com/MilanDroid/

# Requirements
# Numpy, install with: `pip install numpy`
# Tensorflow, install with: `pip install tensorflow`
# Keras, install with: `pip install keras`

# Project version
__version__ = "0.0.1"

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense

# Creamos la matriz con nuestros datos de entrenamiento,
# las 4 combinaciones de las compuertas XOR
training_data = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
], "float32")

# Creamos nuestra matriz de resultados esperados
target_data = np.array([
    [0],
    [1],
    [1],
    [0]
], "float32")

# Se crea un modelo vacio del tipo secuencial,
# esto quiere decir que se creara una serie de neuronas
# de forma secuencial, una delante de otra
model = Sequential()
# Agregamos capas Dense, la capa de entrada con 2 neuronas
# y la capa oculta de 16 neuronas. Con una funcion de activacion 'relu'
# La cual si x >= 0 retorna 'x', si x < 0 retorna '0' 
model.add(Dense(16, input_dim=2, activation='relu'))
# Agregamos una capa con 1 neurona de salida que tiene una funcion 'sigmoid'
# la cual retorna f(x) = 1 / ( 1 + e^(-x) )
model.add(Dense(1, activation='sigmoid'))

# Para entrenar la red se indica el tipo de perdida que utilizaremos,
# en este caso 'optimizador' de los pesos de las conexiones de las neuronas
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['binary_accuracy'])
# Indicamos las entradas y las salidas, ademas de definir la cantidad
# de epocas para el entrenamiento
model.fit(training_data, target_data, epochs=200)

# Evaluamos el modelo, la cantidad de aciertos
scores = model.evaluate(training_data, target_data)

# Imprimimos la cantidad de aciertos que tuvo el modelo entrenado
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# Ahora probamos si el modelo funciona ingresando los datos de entrada
# e imprimimos el vector de resultado
print (model.predict(training_data).round())