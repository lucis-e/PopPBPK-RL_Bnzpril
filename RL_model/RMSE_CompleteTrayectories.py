# Cálculo del valor de RMSE de los valores de los parámetros AUC y Cmax de todas las trayectorias completas obtenidas de las muestras
# de interacción con el modelo PBPK. Se consideran trayectorias completas aquellas que tienen información para 7 días completos, es 
# decir, cuyos estados terminales son en t=7 y la simulación no ha sido parada prematuramente debido a que los valores de ambos pará-
# metros se encontraban fuera de los rangos objetivo.

# Se considera la mejor trayectoria aquella que rinde el RMSE más bajo

#---------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import csv
import docx
from docx.shared import Pt
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
#---------------------------------------------------------------------------------------------------------

# LISTAS

# Columnas de AUC de metabolito 
auc = []    
for i in range(7):
    auc.append('AUC day ' + str(i + 1))

auc_0 = []
for i in range(7):
    auc_0.append("AUC day " + str(i))


# Columnas de Cmax de metabolito
cmx = []   
for i in range(7):
    cmx.append('Cmax day ' + str(i + 1))


# Columnas de dosis administrada
dose = []    
for i in range(7):
    dose.append('Dose day ' + str(i + 1))


# FUNCIONES

TARGET1 = 842    # Valor de AUC objetivo
LB1, UB1, = 474.0, 1210.0  # Límites superior e inferior del intervalo de confianza
TARGET2=175  # Valor de Cmax objetivo
LB2, UB2 = 131.0, 219.0

def RMSE_multi(df):
    eucl_dist = 0     # Distancia planar estandarizada para cada paso de tiempo
    for i, j in zip(auc, cmx):
        value_auc = float(df[i])  
        value_cmx =  float(df[j])  
        eucl_dist += np.sqrt(((value_auc - TARGET1) / (TARGET1 - LB1))**2 + ((value_cmx - TARGET2) / (TARGET2 - LB2))**2)**2  
    return(np.sqrt(eucl_dist / 7))   # Raiz del valor medio de la suma de los errores (distancias euclideas estandarizadas)


action_space = [2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25, 27.5, 30, 32.5, 35, 37.5, 40] 
def Dose_cuantif(dose): # Codificación del espacio de acciones 
    dose = int(dose)
    cuantif = action_space[dose - 1]
    return(cuantif)

# MAIN CODE

# 1. Carga de los datos

input_data = '../Data/A2_DOK_P40GFRstratL_data_V02.xlsx'
input_statemap = 'Resultados/mapa_estados_train.txt'
out_RMSE = 'Resultados/'

antropometric_data = pd.read_excel(input_data, sheet_name = 'APP', keep_default_na=False)

patients = []   # ID de paciente
for i in range(len(antropometric_data)):
    patients.append('p'+str(i+1))

pharmak_cols=[] # Columnas para el resultado farmacocinético (7 days: Dose, Dose previous day, AUC, Cmax, vector 0/1 (dentro de rango o no), stage (1-5))
for i in range(7):
    pharmak_cols += (['Dose day ' + str(i + 1), 'Previous dose day ' + str(i), 'AUC day ' + str(i + 1), 'Cmax day ' + str(i + 1), 'In-range vector '+ str(i + 1), 'Stage ' + str(i + 1)])

pharmak_data = {}  
for i in patients:
    pharmak_data[i] = pd.read_excel(input_data, sheet_name = i, header = None, names = pharmak_cols)

state_map = ''
with open(input_statemap,'r') as f:
         for i in f.readlines():
            state_map=i 
state_map = eval(state_map)


# 2. Extracción de las trayectorias completas

patient_exp={}
for p in patients:    
    complete_exp = pharmak_data[p][['Dose day 1', 'AUC day 1', 'Cmax day 1']][pharmak_data[p]['In-range vector 1'] == 1]  # día 1

    for day in range(2, 8): # días 2-7
        new_day = pharmak_data[p][['Dose day ' + str(day), 'AUC day '+ str(day), 'Cmax day ' + str(day), 'In-range vector ' + str(day)]][~pd.isnull(pharmak_data[p]['AUC day ' + str(day)])]  # Extracción de la información farmacocinética para ese día, eliminando las filas sin información

        rep_num=[] # Cuenta el número de trayectorias que se deben crear a partir de ese nodo medio, es decir, cuantos de los nodos del siguiente día NO son terminales
        for i in range(0, len(new_day), 16):    # Conteo del número de estados que no son terminales por cada bloque de 16 muestras de interacción (que corresponden con las 16 posibles dosis administradas)
            rep_num.append(new_day['In-range vector ' + str(day)].iloc[i:i+16].sum())

        complete_exp = pd.DataFrame(np.repeat(complete_exp.values, rep_num, axis = 0), columns=complete_exp.columns)    # Se repite cada fila el número de veces contadas para el siguiente, es decir, los nodos no terminales que derivan de este nodo medio
        complete_exp = pd.concat([complete_exp, new_day[new_day['In-range vector ' + str(day)] == 1].reset_index(drop=True)], axis = 1).drop('In-range vector ' + str(day), axis = 1)  # junta las filas anteriores con las nuevas, la trayectoria hasta ese momento con el estado en el día siguiente

    patient_exp[p] = np.round(complete_exp, decimals = 3)   # Se redondea el dataset en el orden de los milímetros para estar en concordancia con el aprendizaje


# 3. Cálculo de RMSE y orden de las trayectorias completas en función a su valor 

for i in patients:  
    patient_exp[i]['RMSE'] = patient_exp[i].apply(RMSE_multi, axis = 1) 
    patient_exp[i] = patient_exp[i].sort_values(by='RMSE', ascending = True)   # ordenadas en función del RMSE (a menor RMSE, menor error respecto del punto bidimensional objetivo y por tanto mejor trayectoria)
    patient_exp[i].insert(0, column = 'rank', value = list(range(1, len(patient_exp[i])+1)))


lower_RMSE = pd.DataFrame() # Extracción de la trayectoria con menor RMSE (mejor trayectoria)
for i in patients:
    lower_RMSE = pd.concat([lower_RMSE, patient_exp[i][patient_exp[i]['rank'] == 1]])

lower_RMSE['Patient_id'] = patients
lower_RMSE.to_pickle('Resultados/tray_menor_RMSE')


#with open(out_RMSE,'w') as out:  # Se guarda la información de todas las trayectorias de los pacientes
#    csv_out = csv.writer(out)
#    csv_out.writerows(patient_exp)