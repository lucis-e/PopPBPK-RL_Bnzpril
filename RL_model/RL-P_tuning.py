import pandas as pd
import numpy as np
import os
import csv
import pickle
import random
from gym import Env
from gym.spaces import Discrete
import time
import matplotlib.pyplot as plt
import seaborn as sns
import docx
from docx.shared import Pt


# 0. CARGA DE LOS DATOS

exp_RU='Resultados/experiencias_RU_train.csv'   # Experiencias con RU
exp_RE='Resultados/experiencias_RE_train.csv'   # Experiencias con RE

input2='Resultados/mapa_estados_train.txt'    # Mapa de estados
input_data = "../Data/A2_DOK_P40GFRstratL_data_V02.xlsx"    # Todo el dataset para obtener la información antropométrica de los pacientes

# Carga de las experiencias
data = {}
file = open(exp_RU, 'r')
data_RU = list(csv.reader(file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC))  # lee las experiencias como una lista de listas
file.close()
data_RU = [list(x) for x in set(tuple(x) for x in data_RU)]   # Lista única de experiencias porque solo necesitamos la información 

file = open(exp_RE, 'r')
data_RE = list(csv.reader(file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC))  
file.close()
data_RE = [list(x) for x in set(tuple(x) for x in data_RE)]   # Lista única de experiencias porque solo necesitamos la información 


# Carga de los estados
states = ''
with open(input2,'r') as f:
         for i in f.readlines():
            states=i #string
states = eval(states)

# Carga de las características antropométricas de los pacientes
antropometric_data = pd.read_excel(input_data, sheet_name = 'APP', keep_default_na=False)
antropometric_data = antropometric_data[['Gender', 'BMI', 'GFR']]   # selección de variables que definen los estados del modelo


# 1. FUNCIONES PRINCIPALES

    # Codificación de los estados

def BMI_group(Bmi): 
    if (Bmi < 16):
        subgroup = 0  
    elif (16 <= Bmi < 18.5):
        subgroup = 1    
    elif (18.5 <= Bmi < 25):
        subgroup = 2   
    elif (25 <= Bmi < 30):
        subgroup = 3   
    elif (Bmi >= 30):
        subgroup = 4   
    return (subgroup)


def GFR_group(GFR):
    if (GFR >= 90):
        subgroup = 0
    elif (90 > GFR >= 60):
        subgroup = 1 
    elif (60 > GFR >= 45):
        subgroup = 2
    elif (45 > GFR >= 30):
        subgroup = 3 
    elif (30 > GFR >= 15):
        subgroup = 4 
    return (subgroup)



    # Codificación del espacio de acciones

action_space = [2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25, 27.5, 30, 32.5, 35, 37.5, 40] 

def Dose_cuantif(dose):
   dose = int(dose)
    # Da la dosis en ng/mL (espacio de acciones: [2.5-40]) a partir del código (1-16).
   cuantif = action_space[dose - 1]
   return(cuantif)



    # Función para el cálculo de la distancia euclidiana entre el punto observado (AUCt, Cmaxt) y el objetivo (AUCtarget, Cmaxtarget)

def distance(tuple_state):
    DF = pd.DataFrame([i for i in states if i[0] == tuple_state[0] and i[1] == tuple_state[1] and i[2] == tuple_state[2]], columns=list(antropometric_data.columns) +['AUC', 'Cmax']) # selecciona los estados que tengan las mismas características antropométricas que el paciente seleccionado
    DF['E_dist'] = np.sqrt((DF['AUC'] - tuple_state[3])**2 + (DF['Cmax'] - tuple_state[4])**2) # El AUC está en posición 3 del vector de estados y el Cmax en la 4
    closest_state = DF[DF.E_dist == DF.E_dist.min()].iloc[:, :-1].to_records(index = False)[0]    # select the state with minimun Euclidean distance (dependent on AUC) and extract the state_tuple
    return(states[closest_state.item()])   # returns the codified version of the state



    # Función de estado más próximo: Busca el estado visitado durante el entrenamiento más próximo al estado observado en términos de distancia euclídea del punto formado por las variables farmacocinéticas.

def closest_state(new_st):
    new_state = list(states.keys())[list(states.values()).index(new_st)] # búsqueda de la clave por valor para obtener la tupla de estados completa a partir del indentificador -> para extraer la información farmacocinética del estado
    return(distance(new_state)) # Búsqueda del estado más cercano en términos de distancias planares



    # Cálculo del RMSE de las distancias euclídeas para cada punto.

TARGET1 = 842    
TARGET2=175  
LB1, UB1, = 474.0, 1210.0  
LB2, UB2 = 131.0, 219.0

def Eucl_multi(df):     # Se obtiene la distancia euclidiana
    value_auc = float(df['AUC'])  
    value_cmx =  float(df['Cmax'])  
    eucl_dist = np.sqrt(((value_auc - TARGET1) / (TARGET1 - LB1))**2 + ((value_cmx - TARGET2) / (TARGET2 - LB2))**2)**2  
    return(eucl_dist)   # raíz de la media de la suma de distancias al cuadrado para los 7 días


    # Modificación de datos, cálulo de la recompensa acumulada y RMSE

def f(x):
    d = {}
    d['C_reward'] = x['Reward'].sum()   # cumulative reward
    d['RMSE'] = np.sqrt(x['E_dist'].sum()/7)    # RMSE of the euclidean distances of each observation from the bidimensional target
    return pd.Series(d, index = ['C_reward', 'RMSE'])



# 2. Carga de las tablas-Q

path_UF = 'Resultados/R_utilidad/'  # tablas-Q de la función de utilidad
path_EF = 'Resultados/R_euclidiana/'    # tablas-Q para la función basada en las distancias planares


q_tables_RU = {}    # almacena las tablas-Q con un identificador con la información de los hiperparámetros del modelo

for file in os.listdir(path_UF):
    if (file.endswith('.pickle')) and (file.startswith('qtable')):
        param = file.split(sep = '_')[1:3]  # identificador 
        name = param[0]+ '_' + param[1]
        with open(path_UF+file, 'rb') as fil:
            q_tables_RU[name] = pickle.load(fil)



q_tables_RE = {}    # almacena las tablas-Q con un identificador con la información de los hiperparámetros del modelo

for file in os.listdir(path_EF):
    if (file.endswith('.pickle')) and (file.startswith('qtable')):
        param = file.split(sep = '_')[1:3]  # identificador 
        name = param[0]+ '_' + param[1]
        with open(path_EF+file, 'rb') as fil:
            q_tables_RE[name] = pickle.load(fil)


qtables_names = ['0.8_0.8'] 


# CODIGO PRINCIPAL
# ------------------------------------------------------------------------------------------------------------------------------
# Extracción de las trayectorias de los pacientes siguiendo la política óptima (para cada combinación de hiperparámetros)
# Se utiliza la clase CustomEnv de la libreria gym para python para crear un modelo del entorno personalizado. Este modelo del
# entorno se define por las experiencias observadas generadas a partir de los datos de simulación PBPK.
# Se extraen 40 trayectorias por cada combinación de hiperparámetros
# Estas trayectorias serán posteriormente analizadas para la selección de la mejor combinación de hiperparámetros identificando
# la política óptima. La política óptima es aquella que maximiza la recompensa acumulada descontada, de modo que rinde las trayectorias 
# de mayor recompensa acumulada descontada media.
# ------------------------------------------------------------------------------------------------------------------------------

# Se incializa un primer estado aleatorio necesario para el funcionamiento del entorno personalizado
GENDER = 0
BMI = 0
GFR = 1

tuple_state = (GENDER, BMI, GFR)


# Selección del modelo de función para el que se quiere ajustar los parámetros

#q_tables = q_tables_RU  # Función de utilidad
q_tables = q_tables_RE  # Función dependiente de la distancia planar

# Datos para el modelo del entorno
#data = data_RU
data= data_RE


class CustomEnv(Env):   # Usammos gym para crear el entorno personalizado (el entorno lo definen las experiencias que se generan a partir de los datos obtenidos por el modelo PopPBPK)
    def __init__ (self): # Para inicializar las acciones, los estados y la longitud de los estados
        self.action_space = Discrete(16, start = 1) #   dimensión del espacio de acciones (1-16: 2.5-40 ng)
        self.state_space = Discrete(len(states))    #   dimensión del espacion de estados (10824)
        self.state = states[tuple_state + (0,0)] # Se incializan los estados en AUC = 0 y Cmax = 0, valores de las variables farmacocinéticas pre-tratamiento
        self.treatment_length = 7   # tratamiento para 7 días
        

    def step(self, action):
        # Simulador con modelo de transición dependiente de las experiencias
        experience = random.choice([i for i in data if i[0] == self.state and i[1] == action])   # En cada paso se selecciona la experiencia para el estado en el que se encuentra el agente y para la acción elegida. Como el modeo no es determinista, puede se puede transicionar a varios estados a partir de una misma combinación estado-par, para solventarlo se escoge aleatoriamente una de las experiencias posibles para ese par, simulando el no-determinismo del modelo con los datos disponibles u 
        
        self.state = closest_state(experience[2]) # Transición determinada según la experiencia -> tenemos que calcular la distancia Euclídea (será el mismo estado pero con el modelo PBPK no determinista será uno no visitado)
        reward = experience[3]  # Extraemos el reward 
        self.treatment_length -= 1  # Reducimos en 1 la duración del tratamiento (pasan 24h)
        
        # Checkeamos si la semana se ha acabado
        if self.treatment_length <= 0: 
            done = True
        else:
            done = False

        # Por si queremos añadir algo de información (cómo es el paciente o la experiencia por ejemplo?)
        info = {}
        
        # Returning the step information
        return self.state, reward, done, info
    
    
    def reset(self):
        self.state = states[tuple_state + (0,0)] # Adding the AUC initialized to 0 (initial conditions)
        self.treatment_length = 7
        return self.state



env = CustomEnv()   # Se incializa el entorno



# BLOQUE FINAL: ejecución del programa. Iteración con el entorno personalizado para la extracción de las trayectorias optimizadas
# Cálulo de la recompensa acumulada y selección de mejores hiperparámetros del algorimo de Q-learning

optimal_policies={}


for key in qtables_names:

    qtab = q_tables[key]
    df = pd.DataFrame([])

    for index, row in antropometric_data.iterrows():    # Itera sobre todos los pacientes -> para hacerlo lo más parecido al módulo PBPK voy a llamar a la función desde aquí
        GENDER = row['Gender']
        BMI = BMI_group(row['BMI'])
        GFR = GFR_group(row['GFR'])

        tuple_state = (GENDER, BMI, GFR)

        state = env.reset() # Inicializa el entorno y empieza en el estado inicial pre-tratamiento para el paciente seleccionado
        done = False
        score = 0  
        print(f'Paciente: p{index+1}   Combinación de hiperparámetros (alfa_gamma): {key}')

        action_seq = []
        auc_seq = []
        cmax_seq = []
        reward_seq = []

        while not done:
            action = np.argmax(qtab[state,:]) + 1 # returns the column index (action) with max value of the row (state). Plus one because the q-table index starts on 0 but the dose codification starts on 1
            state, reward, done, info = env.step(action)
            
            action_seq.append(Dose_cuantif(action))
            auc_seq.append(list(states.keys())[list(states.values()).index(state)][3])
            cmax_seq.append(list(states.keys())[list(states.values()).index(state)][4])
            reward_seq.append(reward)
        
        df = pd.concat([df, pd.DataFrame({f'Patient_id':'p'+str(index+ 1), 'Day':list(range(1,8)), 'Dose':action_seq, 'AUC' : auc_seq, 'Cmax':cmax_seq, 'Reward':reward_seq})])
    
    optimal_policies[key] = df


box_data = {}
mean_cr = {}    # Almacena las recompensas acumuladas por cada combinación de hiperparámetros
mean_rmse = {}  # Almacena el RMSE de las trayectorias por cada combinación de hiperparámetros

for key in qtables_names:
    temp_df = optimal_policies[key]
    temp_df['E_dist'] = temp_df.apply(Eucl_multi, axis = 1)     # Distancia euclídea
    temp_df['Patient_id'] = pd.Categorical(temp_df['Patient_id'], temp_df['Patient_id'].unique())
    box_data[key] = temp_df.groupby('Patient_id').apply(f).reset_index()
    mean_cr[key], mean_rmse[key] = box_data[key].mean(numeric_only = True)


mean_cr_df = pd.DataFrame.from_dict(mean_cr, orient = 'index').reset_index()
mean_cr_df.columns = ['index', 'mean_CR']
mean_cr_df[['alpha', 'gamma']] = mean_cr_df['index'].str.split('_', 1, expand=True).astype('float') # Del identificador de hiperparámetros se extraen los valores de alfa y gamma

mean_cr_df = mean_cr_df.sort_values(by = ['alpha', 'gamma']).drop('index', axis = 1)

# Almacena los datos del ajuste de hiperparámetros: Recompensa acumulada media (40 pacientes) y la combinación de hiperparámetros utilizada en el entrenamiento del algoritmo de AR
#mean_cr_df.to_excel('Resultados/Ajuste_hiperparm_RU.xlsx')  # Guarda los valorespara cada combinación de parámetros siguiendo la poítica óptima aprendida durante el entrenamiento
#mean_cr_df.to_excel('Resultados/Ajuste_hiperparm_RE.xlsx') 

max_CR = mean_cr_df[mean_cr_df['mean_CR'] == max(mean_cr_df['mean_CR'])] # Mejores combinaciones según la recompensa acumulada media para todos los pacientes
print(max_CR)   # Muestra en pantalla las mejores combinaciones de parámetros



