from random import random
import numpy as np


def neurona(w, p, b):
    a = np.dot(w, p[:, q]) + b
    if(a >= 0):
        return 1
    else:
        return 0


p = [
    # a  b  c  d  e  f  g
    [1, 1, 1, 1, 1, 1, 0],  # %0
    [0, 1, 1, 0, 0, 0, 0],  # %1
    [1, 1, 0, 1, 1, 0, 1],  # %2
    [1, 1, 1, 1, 0, 0, 1],  # %3
    [0, 1, 1, 0, 0, 1, 1],  # %4
    [1, 0, 1, 1, 0, 1, 1],  # %5
    [1, 0, 1, 1, 1, 1, 1],  # %6
    [1, 1, 1, 0, 0, 0, 0],  # %7
    [1, 1, 1, 1, 1, 1, 1],  # %8
    [1, 1, 1, 1, 0, 1, 1]  # %9
]

p = np.array(p)
p = np.transpose(p)

tpar = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # Numeros pares
tmay5 = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  # Mayores que 5
tprim = [0, 0, 1, 1, 0, 1, 0, 1, 0, 0]  # Numeros Primos


def menu():
    print('Seleccione una opcion: \n',
    '1 Numeros pares\n',
    '2 Numeros mayores que 5\n',
    '3 Numeros primos')


if __name__ == '__main__':
    menu()
    x = input('Seleccione una opcion:')
    t = tpar
    while(True):
        if(x == '2'):
            t = tmay5
            break
        elif(x == '3'):
            t = tprim
            break
        elif(x == '1'):
            break
        else:
            print('opcion no valida')
            menu()
            x = input('Seleccione una opcion:')

    w = 2*np.random.rand(1, 7)-1
    b = 2*random()-1
    e = np.array([None]*10)

    for epocas in range(0, 500):
        for q in range(0, 10):
            e[q] = t[q] - neurona(w, p, b)
            w += e[q] * np.transpose(p[:, q])
            b += e[q]

    print('Error: \n', e)
    print('\nPesos sinapticos: \n', w)
    print('\nPolarizacion: \n', b)
    input('\nPresione enter para salir')
