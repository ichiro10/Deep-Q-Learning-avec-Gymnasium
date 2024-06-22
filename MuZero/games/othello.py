import datetime
import pathlib

import numpy
import torch

from .abstract_game import AbstractGame

import gymnasium as gym
import ale_py
import cv2

# Classe pour l'environnement du jeu Othello
class OthelloWrapper:
    def __init__(self, gameMode):
        # Initialiser l'environnement du jeu
        self.env = gym.make("ALE/Othello-v5", render_mode=None, mode=gameMode, obs_type="grayscale", frameskip=18)
        self.currPosition = (7, 7)
        self.obs = None
        self.info = None
        
        # Initialiser le jeu
        self.board = self.resetGame()
        self.board = self.readBoard()

    # Réinitialiser l'environnement
    def resetGame(self, seed=1337): 
        self.board = numpy.zeros((1, 8, 8))       
        self.obs, self.info = self.env.reset(seed=seed)
        self.currPosition = (7, 7)
        
        return self.board
        
    
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
                self.board[0, j, i] = color
        
        return self.board
    
    # Trouver la nouvelle position du curseur selon le changement du plateau
    def newPosition(self, oldBoard):
        newBoard = self.readBoard()
        
        # Trouver la case qui a eu un ajout de pion
        for i in range(8):
            for j in range(8):
                if oldBoard[0, j, i] == 0 and newBoard[0, j, i] != 0:
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
                #print("up")
                _, _, _, _, _ = self.env.step(2)
                _, _, _, _, _ = self.env.step(0)
        elif targetPosition[0] > self.currPosition[0]:
            for _ in range(targetPosition[0] - self.currPosition[0]):
                #print("down")
                _, _, _, _, _ = self.env.step(5)
                _, _, _, _, _ = self.env.step(0)
                
        # Faire le bon nombre de déplacements horizontaux
        # Action 3 = droite, 4 = gauche
        if targetPosition[1] < self.currPosition[1]:
            for _ in range(self.currPosition[1] - targetPosition[1]):
                #print("left")
                _, _, _, _, _ = self.env.step(4)
                _, _, _, _, _ = self.env.step(0)
        elif targetPosition[1] > self.currPosition[1]:
            for _ in range(targetPosition[1] - self.currPosition[1]):
                #print("right")
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
                    if oldBoard[0, j, i] != self.readBoard()[0, j, i]:
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
                if self.board[0, j, i] == 1:
                    score -= 1
                elif self.board[0, j, i] == 2:
                    score += 1
        return score


class MuZeroConfig:
    def __init__(self):
        # fmt: off
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available



        ### Game
        self.observation_shape = (1, 8, 8)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(64))  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(2))  # List of players. You should only edit the length
        self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = "expert"  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class



        ### Self-Play
        self.num_workers = 1  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = False
        self.max_moves = 9  # Maximum number of moves if game is not finished before
        self.num_simulations = 25  # Number of future moves self-simulated
        self.discount = 1  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.1
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25



        ### Network
        self.network = "resnet"  # "resnet" / "fullyconnected"
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))

        # Residual Network
        self.downsample = False  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 1  # Number of blocks in the ResNet
        self.channels = 16  # Number of channels in the ResNet
        self.reduced_channels_reward = 16  # Number of channels in reward head
        self.reduced_channels_value = 16  # Number of channels in value head
        self.reduced_channels_policy = 16  # Number of channels in policy head
        self.resnet_fc_reward_layers = [8]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [8]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [8]  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 32
        self.fc_representation_layers = []  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [16]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [16]  # Define the hidden layers in the reward network
        self.fc_value_layers = []  # Define the hidden layers in the value network
        self.fc_policy_layers = []  # Define the hidden layers in the policy network



        ### Training
        self.results_path = pathlib.Path(__file__).resolve().parents[1] / "results" / pathlib.Path(__file__).stem / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 1000000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 64  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 10  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.003  # Initial learning rate
        self.lr_decay_rate = 1  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 10000



        ### Replay Buffer
        self.replay_buffer_size = 3000  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 20  # Number of game moves to keep for every batch element
        self.td_steps = 20  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = False



        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it
        # fmt: on

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        return 1


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = OthelloWrapper(0)
        self.env.resetGame()
        self.board = self.env.readBoard()
        self.player = 2
        self.scores = [2, 2]

    def step(self, action):
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        # Si c'est un coup illégal, on pénalise le joueur de 1 et on ne fait rien
        if action not in self.legal_actions():
            return self.get_observation(), -1, False
        
        # S'il n'y a pas de coup possible, on a fini
        if len(self.legal_actions()) == 0:
            return self.get_observation(), 0, True
        
        # Jouer le coup
        row = action // 8
        col = action % 8
        self.env.jouerCoup((row, col))
        
        # Mettre à jour le plateau
        self.player = abs(self.player - 2) + 1
        self.board = self.env.readBoard()
        
        # Mettre à jour les scores
        self.scores = [0, 0]
        for i in range(8):
            for j in range(8):
                if self.board[0, i, j] == 1:
                    self.scores[0] += 1
                elif self.board[0, i, j] == 2:
                    self.scores[1] += 1

        return self.get_observation(), 1, False

    def to_play(self):
        """
        Return the current player.

        Returns:
            The current player, it should be an element of the players list in the config.
        """
        return 0 if self.player == 2 else 2

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.

        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """
        legal = []
        for cell in range(64):
            row = cell // 8
            col = cell % 8            
            
            # Check si la case est déjà occupée
            if self.board[0, row, col] > 0:
                continue
            
            # Check si c'est un coup adjacent à un pion adverse
            illegal = True
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if i == 0 and j == 0:
                        continue
                    if row + i < 0 or row + i >= 8 or col + j < 0 or col + j >= 8:
                        continue
                    if self.board[0, row + i, col + j] == 1:
                        for k in range(1, 8):
                            if row + i * k < 0 or row + i * k >= 8 or col + j * k < 0 or col + j * k >= 8:
                                break
                            if self.board[0, row + i * k, col + j * k] == 0:
                                break
                            if self.board[0, row + i * k, col + j * k] == self.player:
                                illegal = False
                                break
            if illegal:
                continue            
                
            legal.append(cell)
        return legal

    def reset(self):
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        return self.env.resetGame()

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()
        input("Press enter to take a step ")

    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.

        Returns:
            An integer from the action space.
        """
        choice = input(f"Enter the column to play for the player {self.to_play()}: ")
        while choice not in [str(action) for action in self.legal_actions()]:
            choice = input("Enter another column : ")
        return int(choice)

    def expert_agent(self):
        """
        Hard coded agent that MuZero faces to assess his progress in multiplayer games.
        It doesn't influence training

        Returns:
            Action as an integer to take in the current game state
        """
        return self.env.expert_action()

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        return f"Play column {action_number + 1}"

class Othello:
    def __init__(self):
        self.env = OthelloWrapper(0)
        self.env.resetGame()
        self.board = self.env.readBoard()
        self.player = 2
        self.scores = [2, 2]

    def to_play(self):
        return 0 if self.player == 2 else 2

    def reset(self):
        self.env.resetGame()
        self.board = self.env.readBoard()
        self.player = 2
        self.scores = [2, 2]
        return self.env.readBoard()

    def step(self, action):
        # Si c'est un coup illégal, on pénalise le joueur de 1 et on ne fait rien
        if action not in self.legal_actions():
            return self.get_observation(), -1, False
        
        # S'il n'y a pas de coup possible, on a fini
        if len(self.legal_actions()) == 0:
            return self.get_observation(), 0, True
        
        # Jouer le coup
        row = action // 8
        col = action % 8
        self.env.jouerCoup((row, col))
        
        # Mettre à jour le plateau
        self.player = abs(self.player - 2) + 1
        self.board = self.env.readBoard()
        
        # Mettre à jour les scores
        self.scores = [0, 0]
        for i in range(8):
            for j in range(8):
                if self.board[0, i, j] == 1:
                    self.scores[0] += 1
                elif self.board[0, i, j] == 2:
                    self.scores[1] += 1

        return self.get_observation(), 1, False

    def legal_actions(self):
        legal = []
        for cell in range(64):
            row = cell // 8
            col = cell % 8            
            
            # Check si la case est déjà occupée
            if self.board[0, row, col] > 0:
                continue
            
            # Check si c'est un coup adjacent à un pion adverse
            illegal = True
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if i == 0 and j == 0:
                        continue
                    if row + i < 0 or row + i >= 8 or col + j < 0 or col + j >= 8:
                        continue
                    if self.board[0, row + i, col + j] == 1:
                        for k in range(1, 8):
                            if row + i * k < 0 or row + i * k >= 8 or col + j * k < 0 or col + j * k >= 8:
                                break
                            if self.board[0, row + i * k, col + j * k] == 0:
                                break
                            if self.board[0, row + i * k, col + j * k] == self.player:
                                illegal = False
                                break
            if illegal:
                continue            
                
            legal.append(cell)
        return legal

    def have_winner(self):
        # Pas nécessaire pour Othello
        return False

    def expert_action(self):
        # Pas nécessaire pour Othello
        return False

    def render(self):
        # Pas nécessaire pour Othello
        return False
