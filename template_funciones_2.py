import numpy as np
import scipy

from template_funciones import *

# Matriz A de ejemplo
A_ejemplo = np.array([
    [0, 1, 1, 1, 0, 0, 0, 0],
    [1, 0, 1, 1, 0, 0, 0, 0],
    [1, 1, 0, 1, 0, 1, 0, 0],
    [1, 1, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 1, 1, 1],
    [0, 0, 1, 0, 1, 0, 1, 1],
    [0, 0, 0, 0, 1, 1, 0, 1],
    [0, 0, 0, 0, 1, 1, 1, 0]
])

def calcula_L(A):
    # La función recibe la matriz de adyacencia A y calcula la matriz laplaciana
    # Have fun!!
    K = np.diag(np.sum(A, axis=1).astype(float))
    
    L = K-A
    
    return L

def calcula_R(A):
    # La funcion recibe la matriz de adyacencia A y calcula la matriz de modularidad
    # Have fun!!
    n = A.shape[0]
    dobleDeAristas = np.sum(A) # Suma todos los elementos de la matriz de adyacencia.
    vectorGrado = np.sum(A, axis=1) # Suma de cada fila (grado de salida).

    P = np.eye(n)
    
    for i in range(n):
        for j in range(n):
            P[i][j] = (vectorGrado[i]*vectorGrado[j]) / dobleDeAristas
            
    R = A - P
            
    return R

def calcular_S(v):
    s = np.zeros(v.size)

    for i in range(v.size):
        if(v[i] > 0):
            s[i] = 1
        else:
            s[i] = -1

    sColumna = s.reshape(-1, 1)

    return sColumna

def calcula_lambda(L,v):
    # Recibe L y v y retorna el corte asociado
    # Have fun!
    s = calcular_S(v)

    lambdon = 1/4 * (s.T @ L @ s)
        
    return lambdon

def calcula_Q(R,v):
    # La funcion recibe R y s y retorna la modularidad (a menos de un factor 2E)

    s = calcular_S(v)

    Q = 1/2 * (s.T @ R @ s)

    return Q

def calcularInversaConLU(A):
    L, U = calculaLU(A)

    I = np.eye(A.shape[0])
    Y = scipy.linalg.solve_triangular(L, I, lower=True)
    Inv = scipy.linalg.solve_triangular(U, Y, lower=False)

    return Inv

def metpot1(A,tol=1e-8,maxrep=np.inf, seed=None):
   # Recibe una matriz A y calcula su autovalor de mayor módulo, con un error relativo menor a tol y-o haciendo como mucho maxrep repeticiones
   # seed argumento opcional, si no es None, toma seed como semilla. Sirve para garantizar determinismo en caso de que se requiera.
   if not seed is None:
      np.random.seed(seed) #Garantizo que sea deterministica la aleatoriedad
    
   v = np.random.uniform(-1, 1, size=A.shape[0]).reshape(-1, 1) # Generamos un vector de partida aleatorio, entre -1 y 1
   v = v / np.linalg.norm(v) # Lo normalizamos
   v1 = A@v # Aplicamos la matriz una vez
   v1 = v1 / np.linalg.norm(v1)  # normalizamos
   l = (v.T @ A @ v)/ (v.T @ v) # Calculamos el autovector estimado
   l1 = (v1.T @ A @ v1)/ (v1.T @ v1) # Y el estimado en el siguiente paso
   nrep = 0 # Contador
   while np.abs(l1-l)/np.abs(l) > tol and nrep < maxrep: # Si estamos por debajo de la tolerancia buscada 
      v = v1 # actualizamos v y repetimos
      l = l1
      v1 = A@v # Calculo nuevo v1
      v1 = v1 / np.linalg.norm(v1) # Normalizo
      l1 = (v1.T @ A @ v1)/ (v1.T @ v1) # Calculo autovector
      nrep += 1 # Un pasito mas
   if not nrep < maxrep:
      print('MaxRep alcanzado')
   l = (v1.T @ A @ v1)/ (v1.T @ v1) # Calculamos el autovalor
   return v1,l,nrep<maxrep

def deflaciona(A,tol=1e-8,maxrep=np.inf, seed=None):
    # Recibe la matriz A, una tolerancia para el método de la potencia, y un número máximo de repeticiones
    v1,l1,_ = metpot1(A,tol,maxrep, seed=seed) # Buscamos primer autovector con método de la potencia
    deflA = A - l1*(np.outer(v1, v1)) # Sugerencia, usar la funcion outer de numpy
    return deflA

def metpot2(A,v1,l1,tol=1e-8,maxrep=np.inf, seed=None):
   # La funcion aplica el metodo de la potencia para buscar el segundo autovalor de A, suponiendo que sus autovectores son ortogonales
   # v1 y l1 son los primeors autovectores y autovalores de A}
   # Have fun!
   deflA = deflaciona(A, seed=seed)
   return metpot1(deflA,tol,maxrep, seed=seed)


def metpotI(A,mu,tol=1e-8,maxrep=np.inf, seed=None):

    # Retorna el primer autovalor de la inversa de A + mu * I, junto a su autovector y si el método convergió.

    AmuI = calcularInversaConLU(A + mu* np.eye(A.shape[0]))

    return metpot1(AmuI,tol=tol,maxrep=maxrep, seed=seed)

def metpotI2(A,mu,tol=1e-8,maxrep=np.inf, seed=None):
   # Recibe la matriz A, y un valor mu y retorna el segundo autovalor y autovector de la matriz A, 
   # suponiendo que sus autovalores son positivos excepto por el menor que es igual a 0
   # Retorna el segundo autovector, su autovalor, y si el metodo llegó a converger.
   X = A + mu* np.eye(A.shape[0]) # Calculamos la matriz A shifteada en mu
   iX = calcularInversaConLU(X) # La invertimos
   defliX = deflaciona(iX, seed=seed) # La deflacionamos
   v,l,_ =  metpot1(defliX, seed=seed) # Buscamos su segundo autovector
   l = 1/l # Reobtenemos el autovalor correcto
   l -= mu
   return v,l,_


def laplaciano_iterativo(A,niveles,nombres_s=None, seed=None):
    # Recibe una matriz A, una cantidad de niveles sobre los que hacer cortes, y los nombres de los nodos
    # Retorna una lista con conjuntos de nodos representando las comunidades.
    # La función debe, recursivamente, ir realizando cortes y reduciendo en 1 el número de niveles hasta llegar a 0 y retornar.
    if nombres_s is None: # Si no se proveyeron nombres, los asignamos poniendo del 0 al N-1
        nombres_s = range(A.shape[0])
    if A.shape[0] == 1 or niveles == 0: # Si llegamos al último paso, retornamos los nombres en una lista
        return([nombres_s])
    else: # Sino:
        L = calcula_L(A) # Recalculamos el L
        v,l,_ = metpotI2(L, 7, seed=seed) # Encontramos el segundo autovector de L
        # Recortamos A en dos partes, la que está asociada a el signo positivo de v y la que está asociada al negativo
        # Obtenemos los indices para los cuales el autovector v tiene signo positivo y negativo.
        s = calcular_S(v)
        indices_signo_positivo = [i for i, val in enumerate(s) if val == 1]
        indices_signo_negativo = [i for i, val in enumerate(s) if val == -1]
        
        Ap = A[np.ix_(indices_signo_positivo, indices_signo_positivo)] # Asociado al signo positivo
        Am = A[np.ix_(indices_signo_negativo, indices_signo_negativo)] # Asociado al signo negativo
        # DEBUG
        # print("A:")
        # print(A)
        # print(f"s: {s}")
        # print(f"indices_signo_positivo: {indices_signo_positivo}")
        # print(f"indices_signo_negativo: {indices_signo_negativo}")
        # print("Ap:")
        # print(Ap)
        # print("Am:")
        # print(Am)
        
        return(
                laplaciano_iterativo(Ap,niveles-1,
                                     nombres_s=[ni for ni,vi in zip(nombres_s,v) if vi>0],
                                     seed=seed) +
                laplaciano_iterativo(Am,niveles-1,
                                     nombres_s=[ni for ni,vi in zip(nombres_s,v) if vi<0],
                                     seed=seed)
                )        


def modularidad_iterativo(A=None,R=None,nombres_s=None, seed=None):
    # Recibe una matriz A, una matriz R de modularidad, y los nombres de los nodos
    # Retorna una lista con conjuntos de nodos representando las comunidades.

    if A is None and R is None:
        print('Dame una matriz')
        return(np.nan)
    if R is None:
        R = calcula_R(A)
    if nombres_s is None:
        nombres_s = range(R.shape[0])
    # Acá empieza lo bueno
    if R.shape[0] == 1: # Si llegamos al último nivel
        return([nombres_s])
    else:
        v,l,_ = metpot1(R, seed=seed) # Primer autovector y autovalor de R
        v = v.flatten()
        
        # Modularidad Actual:
        Q0 = np.sum(R[v>0,:][:,v>0]) + np.sum(R[v<0,:][:,v<0])
        if Q0<=0 or all(v>0) or all(v<0): # Si la modularidad actual es menor a cero, o no se propone una partición, terminamos
            return([nombres_s])
        else:
            ## Hacemos como con L, pero usando directamente R para poder mantener siempre la misma matriz de modularidad

            # Obtenemos los indices para los cuales el autovector v tiene signo positivo y negativo.
            s = calcular_S(v)
            # print(f"s: {s}")
            indices_signo_positivo = [i for i, val in enumerate(s) if val == 1]
            indices_signo_negativo = [i for i, val in enumerate(s) if val == -1]
            
            Rp = R[np.ix_(indices_signo_positivo, indices_signo_positivo)] # Parte de R asociada a los valores positivos de v
            Rm = R[np.ix_(indices_signo_negativo, indices_signo_negativo)] # Parte asociada a los valores negativos de v
        
            vp,lp,_ = metpot1(Rp, seed=seed)  # autovector principal de Rp
            vm,lm,_ = metpot1(Rm, seed=seed) # autovector principal de Rm
            
            vp = vp.flatten()
            vm = vm.flatten()
        
            # Calculamos el cambio en Q que se produciría al hacer esta partición
            Q1 = 0
            if not all(vp>0) or all(vp<0):
               Q1 = np.sum(Rp[vp>0,:][:,vp>0]) + np.sum(Rp[vp<0,:][:,vp<0])
            if not all(vm>0) or all(vm<0):
                Q1 += np.sum(Rm[vm>0,:][:,vm>0]) + np.sum(Rm[vm<0,:][:,vm<0])
            if Q0 >= Q1: # Si al partir obtuvimos un Q menor, devolvemos la última partición que hicimos
                return([[ni for ni,vi in zip(nombres_s,v) if vi>0],[ni for ni,vi in zip(nombres_s,v) if vi<0]])
            else:
                # Sino, repetimos para los subniveles
                return(
                    modularidad_iterativo(A,Rp,
                                     nombres_s=[ni for ni,vi in zip(nombres_s,v) if vi>0],
                                     seed=seed) +
                    modularidad_iterativo(A,Rm,
                                     nombres_s=[ni for ni,vi in zip(nombres_s,v) if vi<0],
                                     seed=seed)
                )


#print("L:")
#print(calcula_L(A_ejemplo))

# print("----------------------")
#print(modularidad_iterativo(A_ejemplo))
      
# print("R:")
#print(calcula_R(A_ejemplo))

# print("----------------------")
      
#mu = 10
# print(f"mu: {mu}")
# print("METODO I:")
#v, l, b = metpotI(calcula_L(A_ejemplo), mu)
# print("s (deben estar todos en el mismo equipo):")
#s = calcular_S(v)
# print(s) # Todos estan en el mismo equipo.
# print(f"autovalor más grande de (L+mu)^-1: {l}")
# print(f"el autovalor más pequeño de L {(1/l) - mu}")
# print(f"convergio? {b}")

# print(f"lambda (deberia dar 0): {calcula_lambda(calcula_L(A_ejemplo), v)}") #Debe dar 0.

# print("----------------------")

# print("METODO I2:")
#v, l, b = metpotI2(calcula_L(A_ejemplo), mu) 
# print("s (deben estar 1,2,3,4 en un equipo y 5,6,7,8 ern el otro):")
#s = calcular_S(v)
# print(s) #Debe dar que el 1, 2, 3, 4 son de un equipo, y el 5, 6, 7, 8 de otro, no?
# print(f"autovalor de (L+mu)^-1: {l}")
# print(f"segundo autovalor más pequeño de L {l}")
# print(f"convergio? {b}")

# print(f"lambda (deberia dar 2): {calcula_lambda(calcula_L(A_ejemplo), v)}") #Debe dar 2.

# print("Verificamos que efectivamente sean autovalores y autovectores de L, deberían dar practicamente igual:")
# print(f"L * autovector:\n{calcula_L(A_ejemplo)@v}")
# print(f"autovalor * autovector:\n{l*v}")

# print("----------------------")

#for i in range(1, 5):
#    print("")
#    print(f"voy a calcular laplaciano nivel {i}:")
#    print(f"lapalciano nivel {i}: {laplaciano_iterativo(A_ejemplo, i)}")

# print("----------------------")

## OBS: da lo mismo que calcular laplaciano nivel 2 en este caso.
# print(f"voy a calcular por modularidad: {modularidad_iterativo(A_ejemplo)}")

# for i in range(1, 5):
#     print("")
#     print(f"voy a calcular laplaciano nivel {i}:")
#     print(f"lapalciano nivel {i}: {laplaciano_iterativo(A_ejemplo, i)}")

# print("----------------------")

## OBS: da lo mismo que calcular laplaciano nivel 2 en este caso.
# print(f"voy a calcular por modularidad: {modularidad_iterativo(A_ejemplo)}")