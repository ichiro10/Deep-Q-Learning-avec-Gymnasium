import gymnasium as gym
import ale_py
import cv2

class OthelloWrapper:
    def __init__(self, renderMode, gameMode):
        # Initialiser l'environnement du jeu
        self.env = gym.make("ALE/Othello-v5", render_mode=renderMode, mode=gameMode, obs_type="grayscale")
        self.currPosition = (0, 0)
        self.obs = None
        self.info = None

    def resetGame(self, seed=1337):        
        # Réinitialiser l'environnement
        self.obs, self.info = self.env.reset(seed=seed)
        self.currPosition = (0, 0)
        
    def readBoard(self):
        # Lire l'image de l'écran et convertir en image opencv
        img = cv2.cvtColor(self.obs, cv2.COLOR_GRAY2BGR)
        
        # Initialiser le tableau de jeu 8x8
        board = []
        for i in range(8):
            board.append([0] * 8)
        
        # Pour chaque case, identifier la couleur (1 = noir, 2 = blanc, 0 = vide)
        for i in range(8):
            for j in range(8):
                x, y = (22 + 16 * i, 28 + 22 * j)
                color = 0
                if img[y, x][0] < 100:
                    color = 1
                elif img[y, x][0] > 150:
                    color = 2
                board[j][i] = color
        
        return board
    