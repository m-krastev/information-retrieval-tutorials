from torch import nn
from collections import OrderedDict
import torch.nn.functional as F
import torch


# TODO: Implement this!
class LTRModel(nn.Module):
    def __init__(self, num_features, width):
        """
        Initialize LTR model
        Parameters
        ----------
        num_features: int
            number of features 
        """
        super().__init__()
        ### BEGIN SOLUTION
        # Define the layers of the model
        self.layers = nn.Sequential(
            nn.Linear(num_features, width),
            nn.ReLU(),
            nn.Linear(width, 1)
        )
        ### END SOLUTION

    def forward(self, x):
        """
        Takes in an input feature matrix of size (1, N, NUM_FEATURES) and produces the output 
        Arguments
        ----------
            x: Tensor 
        Returns
        -------
            Tensor
        """
        ### BEGIN SOLUTION
        # Forward pass through the layers
        out = self.layers(x)
        ### END SOLUTION
        return out

    

# TODO: Implement this!
class PropLTRModel(LTRModel):
    def forward(self, p, grading=False):
        """
        Takes in the position tensor (dtype:torch.long) of size (1, N), 
        transforms it into a one_hot embedding of size (1, N, layers[0].in_features) and produces the output
        Arguments
        ----------
            x: LongTensor
            grading: bool (default: False) - optional argument, used for grading purposes
        Returns
        -------
            FloatTensor
        """
        ### BEGIN SOLUTION)
        # Get the number of classes
        num_classes = self.layers[0].in_features
        # Transform the position tensor into a one_hot embedding
        x = F.one_hot(p, num_classes).float()
        # Forward pass through the layers
        out = self.layers(x)
        ### END SOLUTION
        if grading:
            return out, {"one_hot": x, "num_classes": num_classes}
        else:
            return out