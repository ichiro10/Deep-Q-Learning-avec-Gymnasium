from atariWrapper import OthelloWrapper as Othello

# Initialiser le jeu Othello
othello = Othello("human", 0)
othello.resetGame()

# Lire le tableau de jeu
board = othello.readBoard()
print(board)