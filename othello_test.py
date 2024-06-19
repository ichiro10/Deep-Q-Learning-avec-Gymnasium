from atariWrapper import OthelloWrapper as Othello
import numpy as np

# Initialiser le jeu Othello
othello = Othello("human", 0)
othello.resetGame()

# Lire le tableau de jeu
board = othello.readBoard()

# Jouer des coups aléatoires
for _ in range(10):
    # Choisir une position aléatoire
    targetPosition = (np.random.randint(8), np.random.randint(8))
    
    # Déplacer le curseur à la position donnée et jouer le coup
    othello.jouerCoup(targetPosition)
    
    print(othello.score())