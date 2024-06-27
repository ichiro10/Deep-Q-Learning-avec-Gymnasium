import gymnasium as gym
import ale_py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from collections import deque
import random

# Configuration de l'environnement
env = gym.make("ALE/Berzerk-v5", render_mode="rgb_array", frameskip=4)

# Création du modèle DQN
def create_dqn_model(input_shape, num_actions):
    model = tf.keras.Sequential([
        layers.Conv2D(32, (8, 8), strides=4, activation='relu', input_shape=input_shape),
        layers.Conv2D(64, (4, 4), strides=2, activation='relu'),
        layers.Conv2D(64, (3, 3), strides=1, activation='relu'),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(num_actions, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00025),
                  loss='huber_loss')
    return model

# Mise en place de la mémoire de replay
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.buffer)

# Définition de la politique d'action
def epsilon_greedy_policy(state, epsilon, num_actions, model):
    if np.random.rand() <= epsilon:
        return np.random.choice(num_actions)
    else:
        q_values = model.predict(np.expand_dims(state, axis=0), verbose=0)
        return np.argmax(q_values[0])

# Fonction de prétraitement de l'état
def preprocess_state(state):
    state = tf.image.rgb_to_grayscale(state)
    state = tf.image.resize(state, [84, 84])
    state = state / 255.0
    return state.numpy()

# Fonction pour vérifier si le personnage a bougé
def has_moved(prev_state, current_state):
    # Comparer les deux états pour détecter un mouvement
    return not np.array_equal(prev_state, current_state)

# Entraînement de l'agent
def train_dqn(env, num_episodes, batch_size):
    input_shape = (84, 84, 1)  # Forme d'entrée après prétraitement des images
    num_actions = env.action_space.n
    dqn_model = create_dqn_model(input_shape, num_actions)
    target_model = create_dqn_model(input_shape, num_actions)
    target_model.set_weights(dqn_model.get_weights())

    replay_buffer = ReplayBuffer(max_size=100000)
    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_decay = 0.995
    gamma = 0.99
    update_target_frequency = 1000

    best_total_reward = -float('inf')  # Initialiser la meilleure récompense

    for episode in range(num_episodes):
        state = env.reset()
        state = preprocess_state(state[0])  # Prétraitement de l'état initial
        prev_state = state
        done = False
        total_reward = 0

        while not done:
            action = epsilon_greedy_policy(state, epsilon, num_actions, dqn_model)
            next_state, reward, done, _, _ = env.step(action)  # Ajout d'une variable supplémentaire
            next_state = preprocess_state(next_state)

            if not has_moved(prev_state, next_state):
                reward -= 1  # Pénaliser si le personnage n'a pas bougé

            replay_buffer.add((state, action, reward, next_state, done))
            state = next_state
            prev_state = state
            total_reward += reward

            if len(replay_buffer) > batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

                target_q_values = target_model.predict(next_states, verbose=0)
                max_target_q_values = np.max(target_q_values, axis=1)
                targets = rewards + (1 - dones) * gamma * max_target_q_values

                with tf.GradientTape() as tape:
                    q_values = dqn_model(states)
                    q_values = tf.reduce_sum(q_values * tf.one_hot(actions, num_actions), axis=1)
                    loss = tf.keras.losses.Huber()(targets, q_values)

                grads = tape.gradient(loss, dqn_model.trainable_variables)
                dqn_model.optimizer.apply_gradients(zip(grads, dqn_model.trainable_variables))

            if episode % update_target_frequency == 0:
                target_model.set_weights(dqn_model.get_weights())

        if total_reward > best_total_reward:
            best_total_reward = total_reward
            dqn_model.save('dqn_model_best.h5')
            print(f"New best model saved with total reward: {best_total_reward}")

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

    return dqn_model

# Entraînement de l'agent
dqn_model = train_dqn(env, num_episodes=500, batch_size=32)

