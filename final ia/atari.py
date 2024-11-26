from ale_py import roms
import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle

# Parámetros del entorno y Q-learning
env = gym.make('SpaceInvaders-v4')
#env = gym.make('SpaceInvaders-v4', render_mode='human')
action_space_size = env.action_space.n
state_space_size = 256  # Mayor granularidad
gamma = 0.99
learning_rate = 0.1
epsilon = 1.0
epsilon_decay = 0.997  # Decaimiento gradual
epsilon_min = 0.1  # Estabilización en exploración
num_episodes = 1000

# Inicialización de la Q-table
Q = np.zeros((state_space_size, action_space_size))

# Función para preprocesar la observación
def preprocess_observation(observation):
    return int(np.mean(observation) // (255 / state_space_size))

# Función para elegir una acción (epsilon-greedy)
def epsilon_greedy(state):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()  # Exploración
    else:
        return np.argmax(Q[state])  # Explotación

# Variables para seguimiento
episode_rewards = []
average_rewards = []
epsilon_history = []

print("INICIANDO ENTRENAMIENTO...")

# Entrenamiento del agente
try:
    for episode in range(num_episodes):
        observation, info = env.reset()
        state = preprocess_observation(observation)
        total_reward = 0
        done = False

        while not done:
            # Elegir una acción
            action = epsilon_greedy(state)

            # Ejecutar la acción
            next_observation, reward, done, truncated, info = env.step(action)
            next_state = preprocess_observation(next_observation)

            # Actualizar la Q-table
            Q[state, action] = Q[state, action] + learning_rate * (
                reward + gamma * np.max(Q[next_state]) - Q[state, action])

            state = next_state
            total_reward += reward

        # Actualizar epsilon (decay)
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        # Reducir tasa de aprendizaje después de episodios clave
        if (episode + 1) % 200 == 0:
            learning_rate *= 0.9

        # Guardar el total de la recompensa de este episodio
        episode_rewards.append(total_reward)
        epsilon_history.append(epsilon)

        # Calcular y guardar el promedio de recompensas cada 100 episodios
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            average_rewards.append(avg_reward)

        # Imprimir promedio cada 10 episodios
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, Avg Reward (last 10): {np.mean(episode_rewards[-10:]):.2f}, Epsilon: {epsilon:.4f}")

except KeyboardInterrupt:
    print("\nEntrenamiento interrumpido por el usuario.")

# Guardar la Q-table en un archivo
with open("tabla_q_stable_space_invaders.pkl", "wb") as f:
    pickle.dump(Q, f)
print("Q-table guardada en 'tabla_q_stable_space_invaders.pkl'.")

# Mostrar gráficos finales
fig, axs = plt.subplots(3, 1, figsize=(12, 12))
plt.subplots_adjust(hspace=0.5)

# Recompensas por episodio
axs[0].plot(episode_rewards, label='Recompensa por episodio', alpha=0.7)
axs[0].set_title('Recompensas en cada episodio')
axs[0].set_xlabel('Episodio')
axs[0].set_ylabel('Recompensa')
axs[0].legend()

# Promedio de recompensas cada 100 episodios
axs[1].plot(range(0, len(average_rewards) * 100, 100), average_rewards, label='Promedio cada 100 episodios', color='orange')
axs[1].set_title('Promedio de recompensas por bloque de 100 episodios')
axs[1].set_xlabel('Episodio')
axs[1].set_ylabel('Recompensa promedio')
axs[1].legend()

# Descendencia de epsilon
axs[2].plot(epsilon_history, label='Epsilon', color='green')
axs[2].set_title('Decaimiento de Epsilon durante el entrenamiento')
axs[2].set_xlabel('Episodio')
axs[2].set_ylabel('Epsilon')
axs[2].legend()

plt.show()

# Mostrar la Q-table
print("Q-table final:")
print(Q)

# Cerrar el entorno
env.close()
