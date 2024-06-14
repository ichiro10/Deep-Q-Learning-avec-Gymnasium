import numpy as np
import gymnasium as gym
import ale_py

import torch
import torch.optim as optim

from AlphaZero.NeuralNetwork import NeuralNetwork
from AlphaZero.MonteCarlo import MCTS

# Hyperparamètres
inputSize = 3
outputSize = 2
hiddenSize = 128

# Initialiser le réseau de neurones
model = NeuralNetwork(inputSize, hiddenSize, outputSize)

# Initialiser l'environnement Blackjack
env = gym.make("ALE/Blackjack-v5", render_mode=None)

# Entrainer le modèle
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train():
    for episode in range(1000):  # Nombre d'épisodes
        print(f"Episode {episode}")
        
        state = env.reset()[0]
        done = False
        states, actions, rewards = [], [], []
        
        frame = 0
        print(f" Frame {frame}")
        while not done:
            frame += 1
            # Afficher le numéro de frame et réécrire la ligne
            print(f" Frame {frame}", end="\r")
            mcts = MCTS(model, c=1.0, nbSimulations=100, env=env, inputSize=inputSize, outputSize=outputSize, hiddenSize=hiddenSize)
            probs = mcts.get_action_probabilities(state)
            action = np.random.choice(outputSize, p=probs)
            next_state, reward, done, truncated, _ = env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            state = next_state
        
        # Calcul de la valeur de retour (return value)
        returns = []
        G = 0
        for r in rewards[::-1]:
            G = r + G
            returns.insert(0, G)
        
        # Mise à jour du modèle
        optimizer.zero_grad()
        loss = compute_loss(states, actions, returns)
        loss.backward()
        optimizer.step()

def compute_loss(states, actions, returns):
    states_tensor = torch.FloatTensor(states)
    actions_tensor = torch.LongTensor(actions)
    returns_tensor = torch.FloatTensor(returns)

    logits = model(states_tensor)
    log_probs = torch.log(logits)
    selected_log_probs = returns_tensor * log_probs[range(len(actions)), actions_tensor]
    loss = -selected_log_probs.mean()
    return loss

train()