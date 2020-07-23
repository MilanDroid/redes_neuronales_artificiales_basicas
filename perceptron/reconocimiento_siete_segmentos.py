#! python

# IA 2020 II PAC, Tarea.
# Redes Neuronales Artificiales básicas para Reconocimiento de Patrones
# Perceptron, reconocimiento de numeros en 7 segmentos

# Video de referencia por Hackeando Tec
# https://www.youtube.com/watch?v=wOWmsDqYx5E

# Made by MilanDroid
# https://github.com/MilanDroid/

# Requirements
# Numpy, install with
# pip install numpy

# Project version
__version__ = "0.0.1"


import numpy as np
import random as random
import matplotlib.pyplot as plt

# Una matriz que representa el estado de los segmenteos
# para representar cada numero
#   ____
#  |   |
#  ----
# |   |
# ----
numbers = [
    #a, b, c, d, e, f, g
    [1, 1, 1, 1, 1, 1, 0], #0
    [0, 1, 1, 0, 0, 0, 0], #1
    [1, 1, 0, 1, 1, 0, 1], #2
    [1, 1, 1, 1, 0, 0, 1], #3
    [0, 1, 1, 0, 0, 1, 1], #4
    [1, 0, 1, 1, 0, 1, 1], #5
    [1, 0, 1, 1, 1, 1, 1], #6
    [1, 1, 1, 0, 0, 0, 0], #7
    [1, 1, 1, 1, 1, 1, 1], #8
    [1, 1, 1, 1, 0, 1, 1], #9
]
# Cantidad de patrones que se usaran
patrones = len(numbers)

# Obtenemos la transpuesta de la matriz
# numbers = np.array(numbers).T

# El vector de salida esperado para el Caso I ( numeros pares)
# tpar
respuestaPar = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
# tmay5
respuestaMayorQueCinco = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
# tprimo
respuestaPrimos = [0, 0, 1, 1, 0, 1, 0, 1, 0, 0]
# Prueba; aqui se asigna cual es la respuesta que probaremos
respuestaEsperada = respuestaMayorQueCinco[:]

# Peso sinaptico
# La funcion np.random.rand(n, m) retorna una matriz de n x m
# con valores entre 0 y 1
# por esta razon se multiplica el resultado por 2 y se le resta 1
# ya que necesitamos valores entre -1 y 1
# de modo que si el elemento es 1; entonces 1 * 2 = 2; 2 - 1 = 1
# si el elemento es 0; entonces 0 * 2 = 0; 0 - 1 = -1 
# Para este caso obtenemos una matriz 1 x 7
W = 2 * np.random.rand(1, 7) - 1
# Polarizacion
# Obtenemos un vector de 1 x 1 con un valor
# entre -1 y 1
b = 2 * np.random.rand(1) - 1

# Matriz de errores
# Creamos una matriz de errores vacia la cual sera de 1 x m;
# donde 'm' es la cantidad de patrones a evaluar.
# Para este caso obtenemos una matrix de 1 x 10,
# obtenemos la transpuesta ya que necesitamos que sea 10 x 1
errores = np.array( [None] * patrones ).T

# Imprimir datos actuales
def imprimirValores():
    print('Errores: ', np.array(errores).T)
    print('Pesos: ', W)
    print('Polarizacion: ', b)

# Procesar valores
def procesar(peso, patron, polarizacion):
    respuesta = np.dot(peso, patron) + polarizacion
    return respuesta

# Funcion de activacion
def hardlim(valor):
    respuesta = 0

    if valor >= 0:
        respuesta = 1

    return respuesta

def graficar(x, y):    
    plt.plot([x, y])
    plt.draw()
    plt.pause(0.0001)
    plt.clf()

# Entrenamiento; aqui se ejecuta el proceso de ajustar los pesos
print('------------------------------ INICIO DE ENTRENAMIENTO --------------------------------')
imprimirValores()
# Epocas
# Cambiar el limite superior en la funcion range(cantidad + 1)
# por la cantidad de epocas que se desean hacer. Tener en cuenta que la cantidad de epocas
# sera cantidad - 1; por ello el '+1' en la funcion
for epoca in range(1, 500 + 1):
    #Decomentar estas lineas para ver el progreso de las epocas
    # print('-' * 40)
    # print('-' * 40)
    # print('Epoca: ', epoca)

    # Por cada patron hacemos una iteracion que van desde 1, hasta n;
    # donde 'n' es la cantidad de patrones disponibles
    for q in range(0, patrones):
        # Se recibe un patron de la forma 1 x n
        # con la transpuesta conseguimos la forma n x 1 o segmentos x digito
        transpuesta = np.array(numbers[q]).T
        respuesta = procesar(W, transpuesta, b)
        errores[q] = respuestaEsperada[q] - hardlim(respuesta)
        W = W + (errores[q] * transpuesta)
        b = b + errores[q]


        # Descomentar para ver el progreso de la red
        # print('*' * 20)
        # print('Digito: ', q)
        # imprimirValores()
print('------------------------------ FIN DE ENTRENAMIENTO --------------------------------')
imprimirValores()

# Para cambiar entre los tipos de validaciones cambiar el valor en la linea 59
# por cualquiera de estas opciones:
# 1- respuestaEsperada = respuestaPar[:]
# 2- respuestaEsperada = respuestaMayorQueCinco[:]
# 3- respuestaEsperada = respuestaPrimos[:]
