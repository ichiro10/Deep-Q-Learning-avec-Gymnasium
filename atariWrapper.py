import gymnasium as gym
import ale_py
import cv2

class OthelloWrapper:
    def __init__(self, renderMode, gameMode):
        # Initialiser l'environnement du jeu
        self.env = gym.make("ALE/Othello-v5", render_mode=renderMode, mode=gameMode, obs_type="grayscale", frameskip=18)
        self.currPosition = (7, 7)
        self.obs = None
        self.info = None
        
        # Initialiser le tableau de jeu 8x8
        self.board = []
        for _ in range(8):
            self.board.append([0] * 8)

    # Réinitialiser l'environnement
    def resetGame(self, seed=1337):        
        self.obs, self.info = self.env.reset(seed=seed)
        self.currPosition = (7, 7)
        
        self.board = []
        for _ in range(8):
            self.board.append([0] * 8)
    
    # Convertir l'image de l'écran en tableau de jeu    
    def readBoard(self):
        # Lire l'image de l'écran et convertir en image opencv
        img = cv2.cvtColor(self.obs, cv2.COLOR_GRAY2BGR)
        
        # Pour chaque case, identifier la couleur (1 = noir, 2 = blanc, 0 = vide)
        for i in range(8):
            for j in range(8):
                x, y = (22 + 16 * i, 28 + 22 * j)
                color = 0
                if img[y, x][0] < 100:
                    color = 1
                elif img[y, x][0] > 150:
                    color = 2
                self.board[j][i] = color
        
        return self.board
    
    # Trouver la nouvelle position du curseur selon le changement du plateau
    def newPosition(self, oldBoard):
        newBoard = self.readBoard()
        
        # Trouver la case qui a eu un ajout de pion
        for i in range(8):
            for j in range(8):
                if oldBoard[j][i] == 0 and newBoard[j][i] != 0:
                    self.currPosition = (j, i)
                    return self.currPosition
        
        return self.currPosition
    
    # Déplacer le curseur à la position donnée et jouer le coup
    def jouerCoup(self, targetPosition):
        oldBoard = self.readBoard().copy()
        
        # Faire le bon nombre de déplacements verticaux
        # Action 2 = haut, 5 = bas
        if targetPosition[0] < self.currPosition[0]:
            for _ in range(self.currPosition[0] - targetPosition[0]):
                print("up")
                _, _, _, _, _ = self.env.step(2)
                _, _, _, _, _ = self.env.step(0)
        elif targetPosition[0] > self.currPosition[0]:
            for _ in range(targetPosition[0] - self.currPosition[0]):
                print("down")
                _, _, _, _, _ = self.env.step(5)
                _, _, _, _, _ = self.env.step(0)
                
        # Faire le bon nombre de déplacements horizontaux
        # Action 3 = droite, 4 = gauche
        if targetPosition[1] < self.currPosition[1]:
            for _ in range(self.currPosition[1] - targetPosition[1]):
                print("left")
                _, _, _, _, _ = self.env.step(4)
                _, _, _, _, _ = self.env.step(0)
        elif targetPosition[1] > self.currPosition[1]:
            for _ in range(targetPosition[1] - self.currPosition[1]):
                print("right")
                _, _, _, _, _ = self.env.step(3)
                _, _, _, _, _ = self.env.step(0)
                
        # La position du curseur est maintenant la cible
        self.currPosition = (targetPosition[0], targetPosition[1])
                
        # Jouer le coup et sortir le nouvel état
        self.obs, reward, terminated, truncated, info = self.env.step(1)
        
        # Action 0 jusqu'à ce que la position du curseur change
        turnProcess = True
        while turnProcess and not terminated:
            self.obs, reward, terminated, truncated, info = self.env.step(0)
            
            # Vérifier si le plateau a changé
            boardChanged = False
            for i in range(8):
                for j in range(8):
                    if oldBoard[j][i] != self.readBoard()[j][i]:
                        boardChanged = True
                        break
                    
            if not boardChanged:
                turnProcess = False
        
        return reward, terminated, truncated, info
    
    # Calculer le score du jeu
    def score(self):
        score = 0
        for i in range(8):
            for j in range(8):
                if self.board[j][i] == 1:
                    score -= 1
                elif self.board[j][i] == 2:
                    score += 1
        return score
        
    