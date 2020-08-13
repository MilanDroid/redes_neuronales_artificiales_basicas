import csv
import random as random
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit, logit

# Variables a usar
archivos_fonts = [
    "data/font1.csv",
    "data/font2.csv",
    "data/font3.csv"
]
archivo_respuestas = [
    "data/respuestas1.csv",
    "data/respuestas2.csv",
    "data/respuestas3.csv",
]
neuronas_ocultas = 30
ep = 1
alfa = 0.001

data_entrenamiento = []
respuestas_esperadas = []

# O(n)
# Carga de los datos de entrenamiento, lectura de los archivos
for archivo_font in archivos_fonts:
    data = []
    with open(archivo_font) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        for row in csv_reader:
            row = np.array(row)
            data.append(row.astype(np.float))
    data_entrenamiento.append(data)

# O(n)
# Carga las repsuestas esperadas desde el archivo
for archivo_respuesta in archivo_respuestas:
    data = []
    with open(archivo_respuesta) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        for row in csv_reader:
            row = list(map(int, row))
            data.append(np.array(row))
    respuestas_esperadas.append(np.array(data).T)

# Iniciando los valores de entrenamiento
# Para cada tipo (a1,a2,a3) hay:
# Teniendo p1, p2, p3, p4, p5, p6, p7
# Y t1, t2, t3, t4, t5, t6, t7
data_entrenamiento[0] = np.array(data_entrenamiento[0]).T
data_entrenamiento[1] = np.array(data_entrenamiento[1]).T
data_entrenamiento[2] = np.array(data_entrenamiento[2]).T
data_entrenamiento = np.concatenate((data_entrenamiento[0], data_entrenamiento[1], data_entrenamiento[2]))
respuestas_esperadas = np.concatenate( (respuestas_esperadas[0], respuestas_esperadas[1], respuestas_esperadas[2]) ).T
# Actualizando la cantidad de ejemplos
cantidad_patrones = len(data_entrenamiento[0])
cantidad_pruebas = len(data_entrenamiento)
cantidad_respuestas = len(respuestas_esperadas)

# Iniciando los pesos por cada patron
W1 = ep * ( 2 * np.random.rand(neuronas_ocultas, cantidad_patrones) - 1 ) # 5x63
b1 = ep * ( 2 * np.random.rand(neuronas_ocultas, 1) - 1 ) # 5x1
W2 = ep * ( 2 * np.random.rand(cantidad_respuestas, neuronas_ocultas) - 1 ) # 21x5
b2 = ep * ( 2 * np.random.rand(cantidad_respuestas, 1) - 1 ) # 21x1

# Impresion de los datos de entrenamiento
# print(f'Data: {np.array(data_entrenamiento).shape}')
# print(f'Respuestas esperadas: {np.array(respuestas_esperadas).shape}')
# print(f'Respuestas esperadas: {respuestas_esperadas}')
# print(f'Cantidad de neuronas ocultas: {neuronas_ocultas}')
# print(f'Cantidad de pruebas: {cantidad_pruebas}')
# print(f'Cantidad de respuestas: {cantidad_respuestas}')
# print(f'Cantidad patrones de prueba: {cantidad_patrones}')
# print(f'Ep: {ep}')
# print(f'Alfa: {alfa}')
# print(f'W1: {W1}')
# print(f'W2: {np.array(W2).shape}')
# print(f'b1: {np.array(b1).shape}')
# print(f'b2: {np.array(b2).shape}')

# Funciones de activacion
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))
 
def sigmoid_derivada(x):
    return sigmoid(x)*(1.0-sigmoid(x))
 
def tanh(x):
    return np.tanh(x)
 
def tanh_derivada(x):
    return 1.0 - x**2

def tansig(x):
    return (2/(1+np.exp(-2*x))-1)

def sig_tanh(x):
    return np.divide( (expit(x) - expit(-x)), (expit(x) + expit(-x)) )

# Se usaran 5 neuronas ya que para identificar las letras necesitamos minimo 3 neuronas
# y para identificar el tipo necesitamos 2 neuronas mas
# Quedando las 3 primers para reconocer el tipo de letra y las otras 2 la fuente
# por ejemplo: A = [1,1,1], B=[1,1,0] y el tipo de fuente asi: Fuente1: [1,1], Fuente2: [1,0]
# quedando: A de la fuente uno A = [1,1,1,1,1] y B de la fuente dos B = [1,1,0,1,0]
epocas = 1
error_cuadratico_medio = np.empty([epocas, 1])
for epoca in range(0, epocas):
    if(epoca%100 == 0):
        print(f'Epoca: {epoca}')

    error_cuadratico = np.zeros([cantidad_pruebas, 1])
    a2 = np.zeros([cantidad_pruebas, cantidad_respuestas, 1])
    for patron in range(0, cantidad_pruebas):
        patron = random.randrange(patron + 1)
        # Respuesta esperada para este patron
        respuesta = respuestas_esperadas[:,patron].reshape([cantidad_respuestas, 1])
        # Rearmando vector de patron
        patron_columna = data_entrenamiento[patron].reshape([cantidad_patrones, 1])
        # Resultante cantidad_neuronas_ocultas x 1
        a1 = sig_tanh(np.dot(W1, patron_columna) + b1)
        # Resultante vector de cantidad_respuestas x 1
        a2[patron] = sig_tanh( np.dot(W2, a1) + b2 )
        # Retropropagacion de las sensibilidades cantidad_respuestas x 1
        e = respuesta - a2[patron]
        # La matriz al cuadrado es la matri por su matriz transpuesta la multiplicamos por la retropropagacion y sacamos la sensibilidad de la capa 2
        # cantidad_respuestas x 1
        s2 = -2 * np.dot(( 1 - np.dot(a2[patron], np.array(a2[patron]).T) ), e)
        # Sensibilidad de la capa 1
        # Resultante un vector de cantidad_neuronas_ocultas x 1
        s1 = np.dot(np.diagflat( np.matrix(1 - np.power(a1, 2)) ), np.dot(W2.T, s2))

        # Actualizacion de los pesos sinapticos y la polarizacion
        W2 = W2 - alfa * s2 * a1.T
        b2 = b2 - alfa * s2
        W1 = W1 - alfa * np.dot( s1, patron_columna.T )
        b1 = b1 - alfa * s1

        # Error cuadratico 
        error_cuadratico[patron] = np.dot(e.T, e)
        print(f'Patron: {patron}')
        print(f'Data: {patron_columna}')
        print(f'Error: {e}')
        print(f'Error cuadratico: {error_cuadratico}')

    
    error_cuadratico_medio[epoca]= np.ndarray.sum(error_cuadratico)/cantidad_pruebas

    
print(f'Finalizado en: {epoca} epocas')
# print(f'Error cuadratico medio: {error_cuadratico_medio}')

# for q in (0, cantidad_pruebas):
#     a1 = sig_tanh(W1*data_entrenamiento[:,q] + b1)
#     print(a1)

# for i=2:10
#     y = [y [i*ones(1,500)]]
# NumeroAciertos = sum(y==iwin)
# PorcentajeAciertos = NumeroAciertos/5000*100

newdata = np.squeeze(error_cuadratico_medio) # Shape is now: (10, 80)
plt.plot(error_cuadratico_medio) # plotting by columns
plt.ylabel('Error')
plt.xlabel('Epocas')
plt.tight_layout()
plt.show()
# plt.plot(range(len(error_cuadratico_medio)), error_cuadratico_medio, color='b')