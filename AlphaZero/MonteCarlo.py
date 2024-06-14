import numpy as np
import gymnasium as gym
import torch

class MCTS:
    def __init__(self, model, c=1.0, nbSimulations=100, env=None, inputSize=3, outputSize=2, hiddenSize=128):
        self.model = model
        self.c_puct = c
        self.nbSimulations = nbSimulations
        self.env = env
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.hiddenSize = hiddenSize
        self.max_depth = 100
        self.Q = {}  # Q-values
        self.N = {}  # Visit counts
        self.P = {}  # Prior probabilities

    def search(self, state):
        print("Searching...")
        for _ in range(self.nbSimulations):
            self._simulate(state, depth=0)
    
    def _simulate(self, state, depth):
        # Afficher "Simulation..." avec un nombre aléatoire de . à la fin
        print("Simulation" + "." * np.random.randint(1, 4), end="\r")
        
        if depth >= self.max_depth:
            return 0  # Return 0 reward if max depth is reached

        state_str = str(state)
        if state_str not in self.P:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            self.P[state_str] = self.model(state_tensor).detach().numpy().flatten()
            self.Q[state_str] = np.zeros(len(self.P[state_str]))
            self.N[state_str] = np.zeros(len(self.P[state_str]))
            return 0

        validActions = list(range(self.outputSize))
        u = self.Q[state_str] + self.c_puct * self.P[state_str] * np.sqrt(np.sum(self.N[state_str])) / (1 + self.N[state_str])
        action = np.argmax(u)
        
        if action not in validActions:
            u[action] = float("-inf")
            action = np.argmax(u)
            if action not in validActions:
                return 0

        next_state, reward, done, truncated, _ = self.env.step(action)
        if done:
            return reward
        
        v = self._simulate(next_state, depth + 1)
        self.Q[state_str][action] = (self.N[state_str][action] * self.Q[state_str][action] + v) / (self.N[state_str][action] + 1)
        self.N[state_str][action] += 1
        return v

    def get_action_probabilities(self, state, temperature=1.0):
        self.search(state)
        state_str = str(state)
        counts = [self.N.get((state_str, a), 0) for a in range(self.outputSize)]
        if temperature == 0:
            best_action = np.argmax(counts)
            probs = [0] * self.outputSize
            probs[best_action] = 1
            return probs
        counts = [x ** (1.0 / temperature) for x in counts]
        total = float(sum(counts))
        
        # Attention à la division par 0
        if total == 0:
            return [1 / self.outputSize] * self.outputSize
        
        return [x / total for x in counts]
