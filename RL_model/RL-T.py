# RL_T: Fase de entrenamiento del modelo de RL

# offline: aprendizaje determinado únicamente por la lista de experiencias (y el mapa de estados complementario) sin interacción con el entorno.
# Se itera sobre una lista de experiencias que empieza en un número inicial de ellas que va incrementando en cada episodio, hasta el último episodio donde se itera sobre el total de experiencias simuladas
# en el "frozen dataset". Es una aproximación offline porque el algoritmo no está conectado con el simulador.

# Input:
#   - Experiences.csv: Lista de experiencias extraida del frozen dataset con los perfiles farmacocinéticos de los pacientes simulados
#   - State_map.txt: Diccionario con la codificación de los estados

# Output:
#   - Q-table: matriz de estados-acciones con el valor de "calidad" de cada pareja calculado según el algoritmo de Q-learning
#   - Gráfica Reward per episode (3): la obtenida con cada modelo y la comparativa

# Métodos:
#   - Q-learning with table: AV-Table con valor Q por cada par estado-acción actualizado según la función de calidad Q.

# Adicional:
#   - Forma en la que se ha generado la gráfica de cumulative reward per episode. En cada episodio se recorre el vector de estados
# y se obtiene la acción que maximiza el reward: si Q == 0 es un estado visitado y el reward es 0. Si Q es != 0 entonces se elige la acción, se encuentra
# la experiencia y se obtiene el nuevo estado y el reward

# -------------------------------------------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm  
import time  # using this to keep track of our saved Q-Tables.
import pickle  # to save/load Q-Tables
import random
# -------------------------------------------------------------------------------------------------------------------------------------------------

plt.style.use('ggplot') # Estilo de gráfica
input_exp='Resultados/experiencias_RE_train.csv'
input_states='Resultados/mapa_estados_train.txt'
out = '../'


# 1. Carga de los datos
file = open(input_exp, 'r')     # importa el módulo con las experiencias
data = list(csv.reader(file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC))  # lee las experiencias como una lista de listas
file.close()   

states = '' # importa el mapa de estados como un diccionario
with open(input_states,'r') as f:
         for i in f.readlines():
            states=i #string
states = eval(states) 


# 2. Módulo de entrenamiento 
#--------------------------------------------------------
# Número total de experiencias: 713584
# Experiencias iniciales: 50000
# Incremento de experiencias: 0
# Número total de epsiodios: 500
#--------------------------------------------------------


# Método Q-learning con tabla AV (solo con la información de las experiencias sin interactuar con el entorno): 25 min
def q_learning_with_table(statemap, experiences, y, lr, num_episodes=500, num_exp=50000):
    q_table = np.zeros((len(statemap), 16))   # Inicializa q-table con 16 columnas (ACTION SPACE) y 10800 filas (STATE SPACE) -> cada fila es el identificador de un estado (0-10799) 
    training_reward = []

    for ep in tqdm(range(num_episodes + 1), desc = f'Training the Q-learning algorithm ...'):    # Genera una barra de progreso
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


# 3. Outputs: almacenamiento de tablas y generación de gráficas
# Guarda la q-table como un fcihero. Luego podremos usarla para exploit el metodo poniendo el nombre del archivo:

q_table, q_tr = q_learning_with_table(states, data, y = 0.4, lr = 0.4)

with open(out + f"holisRE_0.4_0.4.pickle", "wb") as f:
    pickle.dump(q_tr, f)

#with open(out + f"holis.pickle", "wb") as f:
#    pickle.dump(q_tr, f)


fig = plt.figure(figsize = (12,5))
ax = fig.add_subplot(1,1,1)
ax.plot(q_tr)
ax.set_xlabel('Episode')
ax.set_ylabel('Total reward')
plt.savefig(out + 'blablbla.png')   # Reward per episode with Q-learning algorithm
