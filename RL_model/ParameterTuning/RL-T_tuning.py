# RL_T: Fase de entrenamiento del modelo de RL
# 
# PRIMER SCRIPT (offline): aprendizaje determinado únicamente por la lista de experiencias (y el mapa de estados complementario) sin interacción con el entorno.
# Se itera sobre una lista de experiencias que empieza en un número inicial de ellas que va incrementando en cada episodio, hasta el último episodio donde se itera sobre el total de experiencias simuladas
# en el "frozen dataset". Es una aproximación offline porque el algoritmo no está conectado con el simulador.

# Input:
#   - Experiences.csv: Lista de experiencias extraida del frozen dataset con los perfiles farmacocinéticos de los pacientes simulados
#   - State_map.txt: Diccionario con la codificación de los estados

# Output:
#   - Q-table: matriz de estados-acciones con el valor de "calidad" de cada pareja calculado según el algoritmo de Q-learning
#   - N-table: similar a la Q-table pero los valores de calidad se obtienen por la suma acumulada de rewards
#   - Gráfica Reward per episode (3): la obtenida con cada modelo y la comparativa

# Métodos:
#   - Q-learning with table: AV-Table con valor Q por cada par estado-acción actualizado según la función de calidad Q.
#   - Naive sum reward agent: los valores de calidad de los pares se calcula como la suma acumulada de los rewards


# -------------------------------------------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm  
import random  # using this to keep track of our saved Q-Tables.
import pickle  # to save/load Q-Table
# -------------------------------------------------------------------------------------------------------------------------------------------------

plt.style.use('ggplot') # Estilo de gráfica

input_states='../Results/StateMap.txt'
input_exp='../Results/TrainExperiences.csv'
out_HT = '../Results/Train/HyperparameterTuning/'


# 1. Carga de los datos

states = '' # importa el mapa de estados como un diccionario
with open(input_states,'r') as f:
         for i in f.readlines():
            states=i #string
states = eval(states) 

file = open(input_exp, 'r')     # importa el módulo con las experiencias con la función de utilidad
data = list(csv.reader(file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC))  # lee las experiencias como una lista de listas
file.close()


# 2. Módulo de entrenamiento 
#--------------------------------------------------------
# Número total de experiencias: 713584
# Número total de epsiodios: 500
# Experiencias por episodio: 5000
# Selección aleatoria de experiencias de la lista completa
#--------------------------------------------------------

# Método Q-learning con tabla AV (solo con la información de las experiencias sin interactuar con el entorno): 25 min
def q_learning_with_table(statemap, experiences, y, lr, num_episodes=500, num_exp=50000):
    q_table = np.zeros((len(statemap), 16))   # Inicializa q-table con 16 columnas (ACTION SPACE) y 10800 filas (STATE SPACE) -> cada fila es el identificador de un estado (0-10799) 
    training_reward = []

    for ep in tqdm(range(num_episodes + 1), desc = f'Training the Q-learning algorithm ... with alfa {lr} and gamma {y}'):    # Genera una barra de progreso
        random.shuffle(experiences)
        episode_exp = experiences[:num_exp]
        plot_r = 0
        
        for exp in episode_exp: # Itera en la lista de experiencias, extrae el estado, la acción, el nuevo estado y el reward para ir actualizando la q-table con estos valores según la función Q.
            s, a, new_s, r = int(exp[0]), int(exp[1]) - 1, int(exp[2]), exp[3]
            #q_table[s, a] = (1 - lr) * old_value + lr(r + y*np.max(q_table[new_s,:]))
            if (a == np.argmax(q_table[s, :])):
                plot_r += r  
            q_table[s, a] += lr*(r + y*np.max(q_table[new_s,:]) - q_table[s, a]) 

        training_reward.append(plot_r)
    return(q_table, training_reward)


# 3. Ajuste de hiperparámetros
#--------------------------------------------------------
# Se itera sobre los valores de alfa y gamma entre 0 y 1 (escogidos mediante muestreo por cuadrícula, creando una malla regular con separación de 0.1 unidades y escogiendo los valores
# en el vértice superior izquierdo), además se añaden 9 combinaciones adicionales para un valor de constante de gamma=0 (las recompensas futuras no tienen niguna influencia
# y solo las recompensas inmediatas tienen relevancia en la estimación del valor de la función Q). Se entrena el algoritmo con las diferentes combinaciones de parámetros y se obtienen
# las tablas-Q estimadas. Las tablas se usarán más adelante para la predicción, sobre el set de entrenamiento, de los regímenes de dosificación óptimos. Aquellas combinaciones
# que permiten el aprendizaje de la mejor política óptima en términos de RMSE de las trayectorias obtenidas serán las seleccionadas.
#--------------------------------------------------------

lr = np.linspace(0.1,1,10) 
pre_y = np.linspace(0.1,1,10)
y = np.concatenate((np.array(0), pre_y), axis = None)

for gamma in y:
   for alpha in lr:
       q_table, training_reward = q_learning_with_table(states, data, y = gamma, lr = alpha)  # Q-learning 
        
       with open(out_HT + f"qtable40_{alpha}_{gamma}_.pickle", "wb") as f:
           pickle.dump(q_table, f)
        


# y = [0.7, 0.9]
# lr = np.linspace(0.2,1,5) 

# for gamma in y:
#     for alpha in lr:
#         q_table, training_reward_RU = q_learning_with_table(states, data, y = gamma, lr = alpha)  
        
#         with open(out_HT + f"qtable40_{alpha}_{gamma}_.pickle", "wb") as f:
#             pickle.dump(q_table, f)
        
