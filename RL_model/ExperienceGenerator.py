
# Código para la extracción de la información de las muestras de experiencia (interacción con el entorno)
# generadas por el modelo PBPK para la simulación de la administración de Benazepril en pacientes hipertensos
# con insuficiencia renal. 

# Módulo de entrenamiento:

# Entrada: conjunto de datos de entrenamiento de 40 pacientes representativos de los grupos de estratificación
# Salida: 
#   - Mapa de estados: archivo .txt que contiene el diccionario con la codificación del estado farmacocinético
#   de los pacientes. Cada estado a tiempo t se define como una tupla que contiene la información de las siguientes 
#   variables: (Género_t, ICM_t, TFG_t, ABC_t, Cmáx_t).
#   – Información de experiencias: archivo .csv con la información de los eventos de interacción con el entorno. 
#   De acuerdo con la naturaleza markoviana de los estados, cada experiencia o evento contiene la información del 
#   par estado-acción a tiempo t actual, el estado en tiempo t+1 inmediantamente posterior consecuencia de la
#   ejecución de la acción a_t, y la recompensa r_t+1 resultante del evento  (s_t, a_t, s_t+1, r_t+1)

# Consideraciones: se debe seleccionar la función de recompensa que se desea implementar en la resolución del PDM. 
# Se debe especificar la función utilizada del módulo Pyhton "funciones_recompensa.py": util_func si se desea utilizar
# la función de utilidad para una planificación multiobjetivo o euclid_func si se desea utilizar la función basada en 
#la distancia planar cuadrática para una planificación monoobjetivo.

#-----------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import math
import csv
import json
import random
import os
import RewardFunction

# LISTAS

auc = [] # Lista de columnas de AUC met para los 7 días (1-7, sin contar el día inicial)
for i in range(7):
    auc.append('AUC day ' + str(i + 1))


cmx = [] # Lista de columnas de Cmax met para los 7 días (1-7, sin contar el día inicial)
for i in range(7):
    cmx.append('Cmax day ' + str(i + 1))


# FUNCIONES

def BMI_group(Bmi):

    #FUNCIÓN: Codificación de subgrupos en función de ICM

    if (Bmi < 16):
        subgroup = 0   # "Severe thinness"
    elif (16 <= Bmi < 18.5):
        subgroup = 1    # "Mild-moderate thinness"
    elif (18.5 <= Bmi < 25):
        subgroup = 2    # "Normal"
    elif (25 <= Bmi < 30):
        subgroup = 3    # "Overweight"
    elif (Bmi >= 30):
        subgroup = 4    # "Obese"
    return (subgroup)


def GFR_group(GFR):

    #FUNCIÓN: Codificación de subgrupos en función de TFG
    
    if (GFR >= 90):
        subgroup = 0 #"Healthy"
    elif (90 > GFR >= 60):
        subgroup = 1 #"Mild RI (90-60)"
    elif (60 > GFR >= 45):
        subgroup = 2 #"Mild-moderate RI (59-45)"
    elif (45 > GFR >= 30):
        subgroup = 3 #"Moderate RI (44-30)"
    elif (30 > GFR >= 15):
        subgroup = 4 #"Severe RI (29-15)"
    return(subgroup)


# -----
train_directory = "Results/Train"
if not os.path.exists(train_directory):
        os.makedirs(train_directory)
        
test_directory = "Results/Test"
if not os.path.exists(test_directory):
        os.makedirs(test_directory)
# -----

# CÓDIGO PRINCIPAL:
# 1. Carga de los datos farmacocinéticos y antropométricos 

input_data = "../Data/A2_DOK_P40GFRstratL_data_V02.xlsx"
out_map = 'Results/StateMap.txt' # output 1: .txt con las correspondencias de la codificacion de los vectores que definen los estados

# Para el estudio del rendimiento del modelo basado en la función de utilidad:
out_exp = 'Results/TrainExperiences.csv' # output 2: .csv con las experiencias (trayectorias): (S_t, a, S_t+1, r) 


antropometric_data = pd.read_excel(input_data, sheet_name = 'APP', keep_default_na=False)

patients = []   # lista de pacientes
for i in range(1, len(antropometric_data) + 1):
    patients.append('p'+str(i))


pharmak_cols=[] # Lista de columnas del output farmacocinético (6 por día, 7 días = 42 columnas)
for i in range(7):
    pharmak_cols += (['Dose day ' + str(i + 1), 'Previous dose day ' + str(i), 'AUC day ' + str(i + 1), 'Cmax day ' + str(i + 1), 'In-range vector '+ str(i + 1), 'Stage ' + str(i + 1)])


pharmak_data = {}  # carga de los datos farmacocinéticos explorados
for i in patients:
    pharmak_data[i] = pd.read_excel(input_data, sheet_name = i, header = None, names=pharmak_cols)



# 2. Codificación de las variables y definición de los estados

antropometric_data["BMI_group"] = antropometric_data['BMI'].map(BMI_group) # Añade columna de identificador de subgrupo en función a ICM
antropometric_data['GFR_group'] = antropometric_data['GFR'].map(GFR_group) # Columna de identificador de grupo de TFG

cols=['p_no', 'Gender', 'BMI_group', 'GFR_group']   # variables antropométricas seleccionadas en el estudio de predictores
antropo_reduced = antropometric_data[cols]

for i in patients:
    pharmak_data[i].insert(0, column = 'p_no', value = patients.index(i) + 1)
    pharmak_data[i] = np.round(antropo_reduced.merge(pharmak_data[i], how = 'right', on = 'p_no'), decimals = 3)    # se aproximan los parámetros farmacocinéticos en el orden de las milésimas

all_data = pd.concat(pharmak_data, ignore_index=True)



# 3. Categorización y transformación numérica de los vectores de estados

state_vectors = []  # Dos tipos de estado: intermedios/finales (AUC simulado en días 1-7) e iniciales (AUC = 0) 

for i, j in zip(auc, cmx):   #   1. Estados simulados
    state_vectors.append(all_data[['Gender', 'BMI_group', 'GFR_group', i, j]].to_records(index = False).tolist())

flat_list = [item for sublist in state_vectors for item in sublist] 
new_flat_list = [t for t in flat_list if not any(isinstance(n, float) and math.isnan(n) for n in t)]  # Elimina las tuplas con NaN (no hay valor de AUC)
unique_list = list(pd.unique(new_flat_list))  # vectores de estado (simulados) únicos 

antropo_tuple = antropo_reduced[['Gender', 'BMI_group', 'GFR_group']].to_records(index = False).tolist()    # lista con las tuplas de las características antropométricas 
initial_states = list(pd.unique([tupl + (0,0) for tupl in antropo_tuple]))   # 2. Estados iniciales (únicos)

space_state = initial_states + unique_list  # Vectores de estado (ESPACIO DE ESTADOS)
state_map = { k : v for v, k in enumerate(space_state)}     # keys = vectores de estado, value = identificador -> asi para que sea más facil trabajar con el diccionario

with open(out_map,'w+') as f:   # Resultado 1: Mapa de estados (espacio de estados codificado)
     f.write(str(state_map))


# 4. Generación de experiencias

experiences = [] 

for p in patients:
    patient = pharmak_data[p] #  Extraemos los datos de cada paciente individualmente
    in_auc = [0]   # AUC inicial (dia 0): para generar experiencias día 1
    in_cmx = [0]    # Cmax inicial (dia 0): para generar experiencias día 1

    for i in range(1, 8):
        final_table = patient[['Gender', 'BMI_group', 'GFR_group', 'Dose day '+str(i), 'AUC day '+str(i), 'Cmax day '+str(i), 'In-range vector '+str(i)]].dropna()
        final_table['AUC day '+str(i-1)], final_table['Cmax day '+str(i-1)] = np.repeat(in_auc, 16),  np.repeat(in_cmx, 16)  # 4.2. Añadir la columna con los AUC (S_t)
        
        # 4.3. AUCs a partir de los cuales se generarán experiencias el día siguiente (no son estados finales, sino intermedios)
        in_auc = list(final_table['AUC day '+str(i)][final_table['In-range vector '+str(i)] == 1])   # Valores de AUC a partir de los cuales sí hay datos simulados para el día siguiente (y cuyo reward va a ser calculado por la función smooth)
        in_cmx = list(final_table['Cmax day '+str(i)][final_table['In-range vector '+str(i)] == 1])   # Valores de AUC a partir de los cuales sí hay datos simulados para el día siguiente (y cuyo reward va a ser calculado por la función smooth)

        # 4.4. Experiencias:
        state_t = final_table[["Gender", "BMI_group", "GFR_group", 'AUC day ' + str(i-1), 'Cmax day ' + str(i-1)]].to_records(index = False).tolist()
        action = final_table[f'Dose day {i}'].tolist()
        state_t1 = final_table[["Gender", "BMI_group", "GFR_group", 'AUC day ' + str(i), 'Cmax day ' + str(i)]].to_records(index = False).tolist()
        reward = [RewardFunction.util_func(item) if item[3] in in_auc and item [4] in in_cmx else -50 for item in state_t1]   # solo calcula reward para aquellos estados en los que al menos uno de los parámetros está dentro del intervalo de confianza (es decir, para aquellos datos a partir de los cuales se siguen generando datos, si no el programa offline da error en la predicción)
    
        for j in range(len(state_t)):
            exp = state_map[state_t[j]], action[j], state_map[state_t1[j]], reward[j]
            experiences.append(exp)


with open(out_exp,'w') as out:  # Resultado 2: información de las experiencias
    csv_out = csv.writer(out)
    csv_out.writerows(experiences)

    