import numpy as np

from mcts import MCTS
from utils import softmax_normalized


class AlphaZeroMCTS(MCTS):
    def __init__(self, mcts_param, model):
        MCTS.__init__(self, mcts_param)
        self.model = model
        self.config = mcts_param

    def add_noise_to_priors(self, priors):
        noise = np.random.dirichlet([self.config['dir_noise']] * priors.size)
        # complete this method, assuming that epsilon is stored in self.config['dir_epsilon']
        # add noise to priors to ensure that the neural net will let a chance for exploration
        priors = (1 - self.config['dir_epsilon']) * priors + self.config['dir_epsilon'] * noise
        return priors

    def compute_priors_and_value(self, node):
        obs = np.expand_dims(node.obs, axis=0)  # add batch size of 1
        priors, value = self.model.compute_priors_and_value(obs)
        if self.config['add_dirichlet_noise']:
            priors = self.add_noise_to_priors(priors)
        return priors, value

    def compute_action(self, node):
        # Run simulations
        for _ in range(self.params['num_simulations']):
            leaf = node.select()
            if leaf.done:
                value = leaf.reward
            else:
                child_priors, value = self.compute_priors_and_value(leaf)
                leaf.expand(child_priors)
            leaf.backup(value)

        # Compute Tree policy target (TPT): todo: complete the tree policy computation
        # Tree policy is a probability distribution over potential actions
        tree_policy = softmax_normalized(node.child_number_visits, self.params['temperature'])

        # Compute Tree value
        # The tree value for a note St is the expected value of its children Q values
        tree_value = np.sum(node.child_q_value * tree_policy)

        # Choose action according to tree policy
        action = np.random.choice(np.arange(node.action_space_size), p=tree_policy)

        return tree_policy, action, tree_value, node.children[action]