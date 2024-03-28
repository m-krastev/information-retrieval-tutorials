from torch import nn
from collections import OrderedDict


# TODO: Implement this!
class LTRModel(nn.Module):
    def __init__(self, num_features):
        """
        Initialize LTR model
        Parameters
        ----------
        num_features: int
            number of features
        """
        ### BEGIN SOLUTION
        super(LTRModel, self).__init__()
        self.num_features = num_features
        hidden_dim = 10
        self.layers = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        ### END SOLUTION

    def forward(self, x):
        """
        Takes in an input feature matrix of size (N, NUM_FEATURES) and produces the output
        Arguments
        ----------
            x: Tensor
        Returns
        -------
            Tensor
        """
        ### BEGIN SOLUTION
        return self.layers(x)
        ### END SOLUTION
