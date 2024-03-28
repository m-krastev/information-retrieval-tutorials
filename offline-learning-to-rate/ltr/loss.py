from torch.nn import functional as F
import torch


# TODO: Implement this!
def pointwise_loss(output, target):
    """
    Regression loss - returns a single number.
    Make sure to use the MSE loss
    output: (float) tensor, shape - [N, 1]
    target: (float) tensor, shape - [N].
    """
    assert target.dim() == 1
    assert output.size(0) == target.size(0)
    assert output.size(1) == 1
    ### BEGIN SOLUTION
    return F.mse_loss(output, target.view(-1, 1))
    ### END SOLUTION


# TODO: Implement this!
def pairwise_loss(scores, labels):
    """
    Compute and return the pairwise loss *for a single query*. To compute this, compute the loss for each
    ordering in a query, and then return the mean. Use sigma=1.

    For a query, consider all possible ways of comparing 2 document-query pairs.

    Hint: See the next cell for an example which should make it clear how the inputs look like

    scores: tensor of size [N, 1] (the output of a neural network), where N = length of <query, document> pairs
    labels: tensor of size [N], contains the relevance labels

    """
    # if there's only one rating
    if labels.size(0) < 2:
        return None
    ### BEGIN SOLUTION
    sigma = 1.0
    scores_diffs = scores - scores.t()  # [N, 1] - [1, N] =  [N, N]
    S_ij = torch.sign(
        labels.unsqueeze(1) - labels.unsqueeze(0)
    )  # [N, 1] - [1, N] =  [N, N]
    loss_matrix = 0.5 * (1 - S_ij) * scores_diffs + torch.log1p(
        torch.exp(-sigma * scores_diffs)
    )

    # Take only elements above the main diagonal to prevent double counting
    # Not just loss_matrix = loss_matrix.triu(1)?
    mask = torch.triu(torch.ones_like(loss_matrix), diagonal=1)
    loss_matrix = loss_matrix * mask

    # Return the mean of the loss matrix, ignoring the upper triangle and diagonal
    return loss_matrix.sum() / mask.sum()
    # ### END SOLUTION


# TODO: Implement this!
def compute_lambda_i(scores, labels):
    """
    Compute lambda_i (defined in the previous cell). (assume sigma=1.)

    scores: tensor of size [N, 1] (the output of a neural network), where N = length of <query, document> pairs
    labels: tensor of size [N], contains the relevance labels

    return: lambda_i, a tensor of shape: [N, 1]
    """
    ### BEGIN SOLUTION
    sigma = 1.0
    pairwise_diffs = scores - scores.t()  # [N, 1] - [1, N] =  [N, N]
    S_ij = torch.sign(
        labels.unsqueeze(1) - labels.unsqueeze(0)
    )  # [N, 1] - [1, N] =  [N, N]
    exp_term = torch.exp(sigma * pairwise_diffs)
    lambda_ij = sigma * (0.5 * (1 - S_ij) - F.sigmoid(-exp_term))
    lambda_i = lambda_ij.sum(1).view(-1, 1)  # Sum up the columns [N, N] -> [N, 1]
    return lambda_i
    # ### END SOLUTION


def mean_lambda(scores, labels):
    return torch.stack(
        [
            compute_lambda_i(scores, labels).mean(),
            torch.square(compute_lambda_i(scores, labels)).mean(),
        ]
    )


def listwise_loss(scores, labels):
    """
    Compute the LambdaRank loss. (assume sigma=1.)

    scores: tensor of size [N, 1] (the output of a neural network), where N = length of <query, document> pairs
    labels: tensor of size [N], contains the relevance labels

    returns: a tensor of size [N, 1]
    """

    ### BEGIN SOLUTION
    # Copying from above.
    sigma = 1.0
    pairwise_diffs = scores - scores.T
    S_ij = torch.sign(
        labels.unsqueeze(1) - labels.unsqueeze(0)
    )  # [N, 1] - [1, N] =  [N, N]
    lambda_ij = sigma * (0.5 * (1 - S_ij) - F.sigmoid(-sigma * pairwise_diffs))

    # Compute the delta_ndcg
    gain = labels[torch.argsort(scores.flatten(), descending=True)].exp2() - 1
    discount = torch.arange(2, len(labels) + 2, device=labels.device).log2()
    idcg = torch.sum(torch.sort(gain, descending=True).values / discount)

    idx = torch.arange(len(labels), device=labels.device).unsqueeze(1)
    discount = discount.unsqueeze(1)
    delta_ndcg = (
        gain[idx] / discount
        + gain[idx.T] / discount.T
        - gain[idx.T] / discount
        - gain[idx] / discount.T
    )
    delta_ndcg = torch.abs(delta_ndcg / idcg)

    # Multiply the lambda_ij by the delta_ndcg
    lambda_ij *= delta_ndcg
    return torch.sum(lambda_ij, dim=1, keepdim=True)
    ### END SOLUTION


def mean_lambda_list(scores, labels):
    return torch.stack(
        [
            listwise_loss(scores, labels).mean(),
            torch.square(listwise_loss(scores, labels)).mean(),
        ]
    )
