# Módulo de Python para las posibles funciones de recompensa comparadas en el estudio.
# Dos posibles funciones de recompensa:
#   - Planificación multiobjetivo: Función de utilidad (util_func)
#   - Planificación monoobjetivo: Función basada en distancias planares cuadráticas (euclid_func)
#-----------------------------------------------------------------------------------------------------------------

import numpy as np

TARGET1 = 842    # Valor objetivo de ABC  (media de los observados en pacientes sanos)
LB1, UB1, = 474.0, 1210.0  # Límites superior e inferior de los intervalos de confianza
TARGET2=175   # Valor objetivo de Cmáx (media de los observados en pacientes sanos)
LB2, UB2 = 131.0, 219.0# Límites superior e inferior de los intervalos de confianza


# Función triangular dependiente de ABC
def AUC_rew(v):
    r = np.piecewise(v, [v<TARGET1, v>=TARGET1], [lambda v:  (v - LB1)/(TARGET1 - LB1), lambda v: (UB1 - v)/(UB1-TARGET1)]) #definición general de la función triangular
    if (v < LB1) or (v > UB1):
        r = 0.
    return (r)

# Función triangular dependiente de Cmáx
def Cmax_rew(v):
    r = np.piecewise(v, [v<TARGET2, v>=TARGET2], [lambda v:  (v - LB2)/(TARGET2 - LB2), lambda v: (UB2 - v)/(UB2-TARGET2)]) #definición general de la función triangular
    if (v < LB2) or (v > UB2):
        r = 0.
    return (r)


def util_func(state):
    AUC_v = AUC_rew(state[3])   # calculo de la recompensa en función a ABC
    Cmax_v = Cmax_rew (state[4])    # calculo de la recompensa en función a Cmax
    reward = (AUC_v + Cmax_v) * 0.5 # función de utilidad con vector de pesos asociado (0.5, 0.5)
    return(reward)
