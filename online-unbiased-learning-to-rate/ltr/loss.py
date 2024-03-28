import torch
import torch.nn.functional as F

# TODO: Implement this!
def listNet_loss(output, target, grading=False):
    """
    ListNet loss introduced in "Learning to Rank: From Pairwise Approach to Listwise Approach".
    :param output: predictions from the model, shape [1, topk, 1]
    :param target: ground truth labels, shape [1, topk]
    :param eps: epsilon value, used for numerical stability
    :param grading: bool (default = False) - optional argument, used for grading purposes
    :return: loss value, a torch.Tensor
    """
    eps = 1e-10  # epsilon value, use this for numerical stability: add it to probs before taking the logarithm!
    ### BEGIN SOLUTION
    # Compute the softmax of the output
    preds_smax = F.softmax(output, dim=1).squeeze(-1)  # [1, topk]
    # Compute the softmax of the target
    true_smax = F.softmax(target, dim=1)  # [1, topk]
    # Add epsilon to the softmax of the output
    preds_smax = preds_smax + eps
    # Compute the log of the softmax of the output
    preds_log = torch.log(preds_smax)
    # Compute the loss
    loss = -torch.sum(true_smax * preds_log)
    ### END SOLUTION
    if grading:
        return loss, {"preds_smax": preds_smax, "true_smax": true_smax, "preds_log": preds_log}
    else:
        return loss


# TODO: Implement this!
def unbiased_listNet_loss(output, target, propensity, grading=False, clip_low = 0.01, clip_high = 1.0):
    """
    ListNet loss introduced in "Learning to Rank: From Pairwise Approach to Listwise Approach".
    :param output: predictions from the model, shape [1, topk, 1]
    :param target: ground truth labels, shape [1, topk]
    :param propensity: propensity, shape [1, topk] or [topk]
    :param grading: bool (default = False) - optional argument, used for grading purposes
    :return: loss value, a torch.Tensor
    """
    eps = 1e-10 # epsilon value, use this for numerical stability: add it to probs before taking the logarithm!
    
    # The following helps the stability and lower variance
    stable_propensity = propensity.clip(clip_low, clip_high)

    ### BEGIN SOLUTION
    # Compute the softmax of the output
    preds_smax = F.softmax(output, dim=1).squeeze(-1)  # [1, topk]
    # Compute the softmax of the target
    true_smax = F.softmax(target / stable_propensity, dim=1)  # [1, topk]
    # Add epsilon to the softmax of the output
    preds_smax = preds_smax + eps
    # Compute the log of the softmax of the output
    preds_log = torch.log(preds_smax)
    # Compute the loss
    loss = -torch.sum(true_smax * preds_log)
    ### END SOLUTION
    if grading:
        return loss, {"preds_smax": preds_smax, "true_smax": true_smax, "preds_log": preds_log}
    else:
        return loss
