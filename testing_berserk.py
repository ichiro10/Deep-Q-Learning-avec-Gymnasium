import gymnasium as gym
import ale_py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Fonction de prétraitement de l'état
def preprocess_state(state):
    state = tf.image.rgb_to_grayscale(state)
    state = tf.image.resize(state, [84, 84])
    state = state / 255.0
    return state.numpy()

# Définition de la politique d'action
def epsilon_greedy_policy(state, epsilon, num_actions, model):
    if np.random.rand() <= epsilon:
        return np.random.choice(num_actions)
    else:
        q_values = model.predict(np.expand_dims(state, axis=0), verbose=0)
        return np.argmax(q_values[0])

# Charger le modèle sauvegardé
model_path = 'dqn_model_best.h5'
dqn_model = load_model(model_path)

# Configuration de l'environnement
env = gym.make("ALE/Berzerk-v5", render_mode="human", frameskip=4)

# Test du modèle
def test_dqn(env, model, num_episodes):
    num_actions = env.action_space.n

    for episode in range(num_episodes):
        state = env.reset()
        state = preprocess_state(state[0])  # Prétraitement de l'état initial
        done = False
        total_reward = 0

        while not done:
            action = epsilon_greedy_policy(state, epsilon=0.0, num_actions=num_actions, model=model)  # Pas d'exploration pendant les tests
            next_state, reward, done, _, _ = env.step(action)  # Ajout d'une variable supplémentaire
            next_state = preprocess_state(next_state)
            state = next_state
            total_reward += reward

        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

# Nombre d'épisodes de test
num_test_episodes = 10

# Exécution des tests
test_dqn(env, dqn_model, num_test_episodes)
