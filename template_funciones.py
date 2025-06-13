import numpy as np
import scipy

def construye_adyacencia(D,m):
    # Función que construye la matriz de adyacencia del grafo de museos
    # D matriz de distancias, m cantidad de links por nodo
    # Retorna la matriz de adyacencia como un numpy.
    D = D.copy()
    l = [] # Lista para guardar las filas
    for fila in D: # recorriendo las filas, anexamos vectores lógicos
        l.append(fila<=fila[np.argsort(fila)[m]] ) # En realidad, elegimos todos los nodos que estén a una distancia menor o igual a la del m-esimo más cercano
    A = np.asarray(l).astype(int) # Convertimos a entero
    np.fill_diagonal(A,0) # Borramos diagonal para eliminar autolinks
    return(A)

def calculaLU(matriz):
    # matriz es una matriz de NxN
    # Retorna la factorización LU a través de una tupla con dos matrices L y U de NxN.
    # Completar! Have fun
    n = matriz.shape[0]
    m = matriz.copy()  # Una copia porque la ire modificando

    r = np.zeros((1, n))

    L = np.eye(n)  # L = M1^-1 * M2^-1 * ...
    U = matriz  # U = M1 * M2 * ... * matriz

    canonicos = np.eye(n)

    mParcial = np.zeros((n, n))  # Donde estaran M1, M2...
    mParcialInv = np.zeros((n, n))  # Donde estaran M1^-1, M2^-1,...

    for i in range(n-1):

        for j in range(i+1, n):

            r[0][j] = m[j][i] / m[i][i]  # Calculo r

        # Aclaración, r es un vector fila, r.T es un vector columna. canonicos[i] es un vector fila.
        mParcial = np.eye(n) - (r.T @ [canonicos[i]])  # Calculo Mi
        mParcialInv = np.eye(n) + (r.T @ [canonicos[i]])  # Calculo Mi^-1

        m = mParcial @ m  # Actualizo m
        L = L @ mParcialInv
        U = mParcial @ U

        r = np.zeros((1, n))  # Limpio

    return (L, U)


def calcula_matriz_C(A):
    # Función para calcular la matriz de trancisiones C
    # A: Matriz de adyacencia
    # Retorna la matriz C

    ### Calcula inversa de la matriz K, que tiene en su diagonal la suma por filas de A
    # K[i][i] := sumatoria de la fila i-esima de A.
    # Kinv[i][i] := 1/(sumatoria de la fila i-esima de A).
    # y 0 si i!=j.

    # v := np.sum(A, axis=1).astype(float) devuelve array de una dimensión donde la i-esima posición esta la sumatoria de la fila i-esima de A casteada como float.
    # v':= np.reciprocal(v) devuelve un array de una dimensión tal que res[i] = 1/v[i]
    # np.diag(v'), donde v' es una array de una dimensión, genera una matriz donde los elementos de su diagonal son los elementos de v'.
    Kinv = np.diag(np.reciprocal(np.sum(A, axis=1).astype(float)))

    ### Calcula C multiplicando Kinv y A
    C = np.matmul(A.T, Kinv)
    return C


def calcula_pagerank(A,alfa):
    # Función para calcular PageRank usando LU
    # A: Matriz de adyacencia
    # d: coeficientes de damping
    # Retorna: Un vector p con los coeficientes de page rank de cada museo
    C = calcula_matriz_C(A)
    N = A.shape[0]  # Obtenemos el número de museos N a partir de la estructura de la matriz A (A es una matriz cuadrada de nxn).
    M = (N/alfa)*(np.identity(N) - (1 - alfa) * C)
    L, U = calculaLU(M)  # Calculamos descomposición LU a partir de C y d
    b = np.ones(N)  # Vector de 1s, multiplicado por el coeficiente correspondiente usando d y N.
    Up = scipy.linalg.solve_triangular(L,b,lower=True)  # Primera inversión usando L
    p = scipy.linalg.solve_triangular(U,Up)  # Segunda inversión usando U
    return p

def calcula_matriz_C_continua(D):
    # Función para calcular la matriz de trancisiones C
    # A: Matriz de adyacencia
    # Retorna la matriz C en versión continua
    D = D.copy()
    F = 1/D
    np.fill_diagonal(F,0)
    Kinv = np.diag(np.reciprocal(np.sum(F, axis=1).astype(float))) # Calcula inversa de la matriz K, que tiene en su diagonal la suma por filas de F
    C = (Kinv @ F).T # Calcula C multiplicando Kinv y F
    return C

def calcula_B(C,cantidad_de_visitas):
    # Recibe la matriz T de transiciones, y calcula la matriz B que representa la relación entre el total de visitas y el número inicial de visitantes
    # suponiendo que cada visitante realizó cantidad_de_visitas pasos
    # C: Matriz de transiciones
    # cantidad_de_visitas: Cantidad de pasos en la red dado por los visitantes. Indicado como r en el enunciado
    # Retorna: Una matriz B que vincula la cantidad de visitas w con la cantidad de primeras visitas v
    B = np.eye(C.shape[0])
    CTransiciones = C.copy()
    for i in range(cantidad_de_visitas-1):
        B = B + CTransiciones  # Sumamos las matrices de transición para cada cantidad de pasos
        CTransiciones = CTransiciones @ C
    return B


def norma_1(matriz):
  # Norma 1 es el máximo de las sumas de los valores absolutos de cada columna.
  return np.max(np.sum(np.abs(matriz), axis=0))
