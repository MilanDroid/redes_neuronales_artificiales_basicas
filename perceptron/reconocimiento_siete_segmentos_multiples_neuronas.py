#! python

# IA 2020 II PAC, Tarea II.
# Redes Neuronales Artificiales básicas para Reconocimiento de Patrones
# Perceptron, reconocimiento de numeros en 7 segmentos,
# Multiples neuronas

# Video de referencia por Hackeando Tec
# https://www.youtube.com/watch?v=wOWmsDqYx5E

# Made by MilanDroid
# https://github.com/MilanDroid/

# Requirements
# Numpy, install with
# pip install numpy

# Project version
__version__ = "0.1.1"


import numpy as np
import random as random
# import matplotlib.pyplot as plt

# Una matriz que representa el estado de los segmenteos
# para representar cada numero
#   ----
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

# Los vectores de salida esperados
respuestaEsperada = [
    # tpar
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    # tmay5
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    # tprimo
    [0, 0, 1, 1, 0, 1, 0, 1, 0, 0]
]

# Peso sinaptico
# La funcion np.random.rand(n, m) retorna una matriz de n x m
# con valores entre 0 y 1
# por esta razon se multiplica el resultado por 2 y se le resta 1
# ya que necesitamos valores entre -1 y 1
# de modo que si el elemento es 1; entonces 1 * 2 = 2; 2 - 1 = 1
# si el elemento es 0; entonces 0 * 2 = 0; 0 - 1 = -1 
# Para este caso obtenemos una matriz 3 x 7
W = 2 * np.random.rand(3, 7) - 1
# Polarizacion
# Obtenemos una matriz de 3 x 1 con valores
# entre -1 y 1
b = 2 * np.random.rand(3, 1) - 1

# Matriz de errores
# Creamos una matriz de errores vacia la cual sera de 1 x m;
# donde 'm' es la cantidad de patrones a evaluar.
# Para este caso obtenemos una matrix de 3 x 10,
# obtenemos la transpuesta ya que necesitamos que sea 10 x 3
errores = [
    np.array( [None] * patrones ).T,
    np.array( [None] * patrones ).T,
    np.array( [None] * patrones ).T
]

# Imprimir datos actuales
def imprimirValores():
    print('Errores: ', np.array(errores).T)
    # print('Respuesta esperada: ', respuestaEsperada)
    print('Pesos: ', W)
    print('Polarizacion: ', b)

# Funcion de activacion
def hardlim(valor):
    respuesta = 0
    if valor >= 0:
        respuesta = 1
    return respuesta

# Procesar valores
def procesar(peso, patron, polarizacion):
    respuesta = np.dot(peso, patron) + polarizacion
    return respuesta

def neurona(digito, tarea):
    transpuesta = np.array(numbers[digito]).T
    resultado = procesar(W[tarea], transpuesta, b[tarea])

    # print('Error: ', errores[tarea][digito])
    # print('Peso: ', W[tarea])
    # print('Polarizacion: ', b[tarea])

    errores[tarea][digito] = respuestaEsperada[tarea][digito] - hardlim(resultado)
    W[tarea] = W[tarea] + (errores[tarea][digito] * transpuesta)
    b[tarea] = b[tarea] + errores[tarea][digito]

# Comprobamos que el resultado es correcto
# con la sumatoria de multiplicar el valor de cada segmento por su peso
# y sumar la polarizacion o 'bias'
def comprobacion():
    resultados = [
        np.array( [None] * patrones ),
        np.array( [None] * patrones ),
        np.array( [None] * patrones )
    ]
    
    for q in range(0, patrones):
        for i in range(0, len(W) ):
            resultados[i][q] = W[i][0] * numbers[q][0]
            resultados[i][q] = resultados[i][q] + W[i][1] * numbers[q][1]
            resultados[i][q] = resultados[i][q] + W[i][2] * numbers[q][2]
            resultados[i][q] = resultados[i][q] + W[i][3] * numbers[q][3]
            resultados[i][q] = resultados[i][q] + W[i][4] * numbers[q][4]
            resultados[i][q] = resultados[i][q] + W[i][5] * numbers[q][5]
            resultados[i][q] = resultados[i][q] + W[i][6] * numbers[q][6]
            resultados[i][q] = resultados[i][q] + b[i]

            resultados[i][q] = hardlim(resultados[i][q])
    print('Comprobacion: ')
    print(resultados[0])
    print(resultados[1])
    print(resultados[2])


def graficar():
    line = np.linspace(0, 9, 100)
    for q in range(0, patrones):
        plt.plot(q, numbers[q][0]*0.1, 'bo') # a
        plt.plot(q, numbers[q][1]*0.2, 'gD') # b
        plt.plot(q, numbers[q][2]*0.3, 'r*') # c
        plt.plot(q, numbers[q][3]*0.4, 'cs') # d
        plt.plot(q, numbers[q][4]*0.5, 'm>') # e
        plt.plot(q, numbers[q][5]*0.6, 'y1') # f
        plt.plot(q, numbers[q][6]*0.7, 'k2') # g

    # plt.plot(line, -b[1]/W[1])    
    plt.xlabel('Digito')
    plt.ylabel('Segmentos')
    plt.title('Numeros pares')
    plt.show()


# Imprimiendo valores de prueba
# for i in range(0, len(errores) ):
#     print('Indice ', i, ': ', errores[i])

# Entrenamiento
print('------------------------------ INICIO DE ENTRENAMIENTO --------------------------------')
imprimirValores()

# Epocas
# Cambiar 'cantidad' para alargar o acortar el entrenamniento
# cada iteracion es una epoca
# range(n) no incluye el limite superior, por ello se suma '1',
# quedando range(n + 1)
for epoca in range(200 + 1):
    # Decomentar estas lineas para ver el progreso de las epocas
    # print('-' * 40)
    # print('-' * 40)
    # print('Epoca: ', epoca)

    # Por cada patron de digito (0-9) hacemos una iteracion que van desde 0, hasta n;
    # donde 'n' es la cantidad de patrones disponibles
    for q in range(0, patrones):
        for i in range(0, len(respuestaEsperada)):
            neurona(q, i)

            # Descomentar para ver el progreso de la red
            # print('*' * 20)
            # print('Digito: ', q)
            # print('Tarea: ', i)
            # imprimirValores()
print('------------------------------ FIN DE ENTRENAMIENTO --------------------------------')
imprimirValores()
comprobacion()
graficar()
